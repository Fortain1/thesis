import random
import datetime
import os
import sys
import ast
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
import gymnasium as gym
from obstacle_env import QuadXObstacleEnv

DEFAULT_HIDDEN_UNITS = 64

def tensor(x, device="cpu"):
    if torch.is_tensor(x):
        return x
    x = np.asarray(x, dtype=np.float32)
    x = torch.tensor(x, device=torch.device(device), dtype=torch.float32)
    return x

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.weight.data.mul_(1)
        nn.init.constant_(m.bias.data, 0)

class DACNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, num_options, device="cuda:0"):
        super(DACNetwork, self).__init__()

        self.higher_net = MasterNetwork(obs_dim, num_options)
        self.lower_nets = nn.ModuleList([LowerNetwork(obs_dim, action_dim) for _ in range(num_options)])
        if not torch.cuda.is_available():
            device = "cpu"
        print("using device: %s" % device)
        self.device = device
        self.apply(init_weights)
        self.to(torch.device(self.device))

    def forward(self, x):
        x = tensor(x, self.device)
        mean = []
        std = []
        beta = []
        for lower_net in self.lower_nets:
            option_pred = lower_net(x)
            mean.append(option_pred["mean_action"].unsqueeze(1))
            std.append(option_pred["std_action"].unsqueeze(1))
            beta.append(option_pred["termination_prob"])
        mean = torch.cat(mean, dim=1)
        std = torch.cat(std, dim=1)
        beta = torch.cat(beta, dim=1)

        master_pred = self.higher_net(x)

        return {
            "mean": mean,
            "std": std,
            "beta": beta,
            "q_option": master_pred["q_option"],
            "master_policy": master_pred["master_policy"],
        }


class MasterNetwork(nn.Module):
    def __init__(self, obs_dim, num_options):
        super(MasterNetwork, self).__init__()

        self.master_policy_net = FCNetwork(obs_dim, num_options, lambda: nn.Softmax(dim=-1))
        self.value_net = FCNetwork(obs_dim, num_options)

    def forward(self, x):
        master_policy = self.master_policy_net(x)
        q_option = self.value_net(x)

        return {
            "master_policy": master_policy,
            "q_option": q_option,
        }


class LowerNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(LowerNetwork, self).__init__()

        self.policy_net = FCNetwork(obs_dim, action_dim, nn.Tanh)
        self.termination_net = FCNetwork(obs_dim, 1, nn.Sigmoid)
        self.std = nn.Parameter(torch.zeros((1, action_dim)))

    def forward(self, x):
        mean_action = self.policy_net(x)
        std_action = F.softplus(self.std).expand(mean_action.size(0), -1) # ?
        termination_prob = self.termination_net(x)

        return {
            "mean_action": mean_action,
            "std_action": std_action,
            "termination_prob": termination_prob,
        }


class FCNetwork(nn.Module):
    def __init__(self,
        input_dim, output_dim, output_activation=None,
        hidden_dims=(DEFAULT_HIDDEN_UNITS, DEFAULT_HIDDEN_UNITS), hidden_activation=nn.Tanh
    ):
        super(FCNetwork, self).__init__()

        layers = list()
        dims = (input_dim,) + hidden_dims
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_features=in_dim, out_features=out_dim))
            layers.append(hidden_activation())
        layers.append(nn.Linear(in_features=hidden_dims[-1], out_features=output_dim))
        if output_activation is not None:
            layers.append(output_activation())

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        data = tensor(x)
        for layer in self.layers:
            data = layer(data)

        return data
    
def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step +
                                               1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


def ppo_iter(mini_batch_size, states, actions, log_probs, rets, advs, beta_advs, qos, betas, entropies, options, prev_options, is_init_states):
    batch_size = states.size(0)
    idlist = np.random.permutation(batch_size)
    for i in range(batch_size // mini_batch_size):
#        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        rand_ids = idlist[i*mini_batch_size:min((i+1)*mini_batch_size, batch_size)]
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[
            rand_ids, :], rets[rand_ids, :], advs[rand_ids, :], beta_advs[rand_ids, :], qos[rand_ids, :], betas[rand_ids, :], entropies[rand_ids, :], options[rand_ids, :], prev_options[rand_ids, :], is_init_states[rand_ids, :]

# states.shape is [num_envs*mini_batch_size, observation_space]
# dist.log_prob(action) gives a tensor with shape [num_envs*mini_batch_size, action_space],
# thus needs to use .sum(1).unsqueeze(1) to transform to [num_envs*mini_batch_size, 1]
def ppo_update(model,
               optimizer,
               ppo_epochs,
               mini_batch_size,
               states,
               actions,
               log_probs,
rets, advs, beta_advs, qos, betas, entropies, options, prev_options, is_init_states,
               clip_param=0.2):
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, ret, adv, beta_adv, qo, beta, entropy, option, prev_option, is_init_state in ppo_iter(
                mini_batch_size, states, actions, log_probs, rets, advs, beta_advs, qos, betas, entropies, options, prev_options, is_init_states):
            prediction = model(state)
            option_ = option.unsqueeze(-1).expand(-1, -1, prediction['mean'].size(-1))
            mean = prediction['mean'].gather(1, option_).squeeze(1)
            std = prediction['std'].gather(1, option_).squeeze(1)
            dist = torch.distributions.Normal(mean, std)
            #            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action).sum(-1).unsqueeze(-1)
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_param,
                                1.0 + clip_param) * adv

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (ret - prediction['q_option'].gather(1, option)).pow(2).mean()
            beta_loss = (beta.gather(1, prev_option) * beta_adv * (1 - is_init_state)).mean()
            master_loss = -(prediction['master_policy'].gather(1, option) * adv).mean() - 0.01 * (-(prediction['master_policy']*prediction['master_policy'].log()).sum(-1).mean())#entropy
            loss = 0.5 * critic_loss + actor_loss  + beta_loss + master_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()


class oc_agent():
    def __init__(self,
                 num_envs=16,
                 env_name="HalfCheetah-v2",
                 lr=3e-4,
                 num_steps=2048,
                 num_options=4,
                 ppo_epochs=10,
                 mini_batch_size=64):
        self.num_envs = num_envs
        self.env_name = env_name
        self.lr = lr
        self.num_steps = num_steps
        self.envs = make_vec_env(env_name, n_envs=self.num_envs)
        self.envs = VecNormalize(self.envs, norm_reward=False)
        self.num_inputs = self.envs.observation_space.shape[0]
        self.num_outputs = self.envs.action_space.shape[0]
        self.num_options = num_options
        self.model = DACNetwork(self.num_inputs, self.num_outputs, self.num_options)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.lr,
                                    eps=1e-5)
        self.max_frames = 2000000
        self.is_init_state = tensor(np.ones((self.num_envs))).byte()
        self.worker_index = tensor(np.arange(self.num_envs)).long()
        self.prev_option = tensor(np.zeros(self.num_envs)).long()
        self.eps = 0.1
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

    def compute_pi_h(self, prediction, pre_option, is_init_states):
        master_policy = prediction["master_policy"]
        beta = prediction["beta"]

        mask = torch.zeros_like(master_policy)
        mask[self.worker_index, pre_option] = 1

        # pi_h = beta * master_policy + (1 - beta) * mask
        is_init_states = is_init_states.view(-1, 1).expand(-1, master_policy.size(1))
        pi_h = torch.where(is_init_states, master_policy, beta * master_policy + (1 - beta) * mask)
        # print("pi_h %s" % pi_h)

        return pi_h

    def compute_pi_l(self, options, action, mean, std):
        options = options.unsqueeze(-1).expand(-1, -1, mean.size(-1))
        mean = mean.gather(1, options).squeeze(1)
        std = std.gather(1, options).squeeze(1)
        normal_dis = torch.distributions.Normal(mean, std)

        pi_l = normal_dis.log_prob(action).sum(-1).exp().unsqueeze(-1)

        return pi_l

    def run(self):
        frame_idx = 0
        test_rewards = []
        state = self.envs.reset()
        cumu_rewd = np.zeros(self.num_envs)
        path='./data/{}'.format(self.env_name)
        try:
            os.mkdir(path)
        except OSError as error:
            print(error)
        curtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S") \
                    + "_{:04d}".format(random.randint(1,9999))
        fd_train = open(path + '/ppoc_train_{}.log'.format(curtime), 'w')
        while frame_idx < self.max_frames:
            log_probs = []

            states = []
            actions = []
            rewards = []
            masks = []
            np_masks = []
            entropies = []
            values = []
            #            entropy = 0
            prev_options = []
            rets = []
            advs = []
            predictions = []
            beta_advs = []
            qos = []
            options = []
            betas = []
            is_init_states = []
            for _ in range(self.num_steps):
                state = torch.FloatTensor(state)
                prediction = self.model(state)
                pi_h = self.compute_pi_h(prediction, self.prev_option, self.is_init_state)
                option = torch.distributions.Categorical(probs = pi_h).sample()
#                option = self.sample_option(prediction, self.eps, self.prev_option, self.is_init_state)
                mean = prediction['mean'][self.worker_index, option]
                std = prediction['std'][self.worker_index, option]
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()

                log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
                entropy = dist.entropy().sum(-1).unsqueeze(-1)



                next_state, reward, done, info = self.envs.step(action.cpu().detach().numpy())
                #done = done or truncated
                cumu_rewd += reward
                for i in range(self.num_envs):
                    if done[i]:
                        print("Cumulative reward at step " + str(frame_idx) +
                              " is " + str(cumu_rewd[i]))
                        fd_train.write("%d %f\n" % (frame_idx, cumu_rewd[i]))
                        cumu_rewd[i] = 0

                    fd_train.flush()

                log_probs.append(log_prob)
                prev_options.append(self.prev_option.unsqueeze(-1))
                options.append(option.unsqueeze(-1))
                rewards.append(torch.FloatTensor(reward).unsqueeze(-1))
                masks.append(torch.FloatTensor(1 - done).unsqueeze(-1))
                entropies.append(entropy)
                states.append(state)
                values.append(prediction['q_option'][self.worker_index, option].unsqueeze(-1))
                actions.append(action)
                betas.append(prediction['beta'])
                qos.append(prediction['q_option'])
                is_init_states.append(self.is_init_state.unsqueeze(-1).float())
                self.is_init_state = tensor(done).byte()
                self.prev_option = option
                state = next_state
                frame_idx += self.num_envs

            next_state = torch.FloatTensor(next_state)
            with torch.no_grad():
                prediction = self.model(next_state)
                betass = prediction['beta'][self.worker_index, self.prev_option]
                ret = (1 - betass) * prediction['q_option'][self.worker_index, self.prev_option] + \
                  betass * torch.max(prediction['q_option'], dim=-1)[0]
                ret = ret.unsqueeze(-1)

            for i in reversed(range(self.num_steps)):
#                v = qos[i].max(dim=-1, keepdim=True)[0] * (1 - self.eps) + qos[i].mean(-1).unsqueeze(-1) * self.eps
                v = (prediction['q_option'] * prediction['master_policy']).sum(-1).unsqueeze(-1)
                q = qos[i].gather(1, prev_options[i])
                beta_advs.append(q - v + 0.01)
            rets = compute_gae(ret, rewards, masks, values)

            log_probs = torch.cat(log_probs).detach()
            betas = torch.cat(betas).detach()
            qos = torch.cat(qos).detach()
            states = torch.cat(states).detach()
            actions = torch.cat(actions).detach()
            rets = torch.cat(rets).detach()
            values = torch.cat(values).detach()
            advs = rets - values
            advs = (advs-advs.mean())/advs.std()
            beta_advs = torch.cat(beta_advs).detach()
            entropies = torch.cat(entropies).detach()
            options = torch.cat(options).detach()
            prev_options = torch.cat(prev_options).detach()
            is_init_states = torch.cat(is_init_states).detach()

            ppo_update(self.model, self.optimizer, self.ppo_epochs,
                       self.mini_batch_size, states, actions, log_probs,
                       rets, advs, beta_advs, qos, betas, entropies, options, prev_options, is_init_states)
        fd_train.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run PPOC on a specific game.')
    parser.add_argument('-e', '--env_name', type=str, help='Name of the game', default="PyFlyt/QuadX-Hover-v1")
    parser.add_argument('-n', '--num_envs', type=int, help='Number of workers', default=1)
    args = parser.parse_args()
    ocagent = oc_agent(num_envs=args.num_envs, env_name=args.env_name)
    ocagent.run()