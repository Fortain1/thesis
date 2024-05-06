import torch
import torch.nn as nn
import torch.functional as F
from collections import deque

import numpy as np
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
    
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.weight.data.mul_(1)
        nn.init.constant_(m.bias.data, 0)

class RolloutBuffer:
    def __init__(self, batch_size, buffer_size) -> None:
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.experiences = {
            "states": deque(maxlen=self.buffer_size),
            "actions": deque(maxlen=self.buffer_size),
            "rewards": deque(maxlen=self.buffer_size),
            "log_probs": deque(maxlen=self.buffer_size),
            "option_values": deque(maxlen=self.buffer_size),
            "betas": deque(maxlen=self.buffer_size),
            "options": deque(maxlen=self.buffer_size),
            "previous_options": deque(maxlen=self.buffer_size),
            "is_init": deque(maxlen=self.buffer_size),
            "masks": deque(maxlen=self.buffer_size),
            "advantages" : deque(maxlen=self.buffer_size),
            "beta_advantes": deque(maxlen=self.buffer_size)
        }
        
    
    def store(self, experience):
        for key, value in experience.items():
            self.experiences[key].append(value)

    def reset(self):
        for key in self.experiences.keys():
            self.experiences[key] = deque(maxlen=self.buffer_size)
    
    def get_experiences(self, keys):
        return tuple(self.experiences.get(key) for key in keys)

    def __iter__(self):
        current_length = len(self.experiences["actions"])
        completed_buffer_length = current_length - current_length % self.batch_size

        ids = np.random.permutation(completed_buffer_length)
        ids_per_batch = np.split(ids, len(ids)//self.batch_size)

        for ids_in_batch in ids_per_batch:
            experiences = self.experiences[ids_in_batch]
            yield tuple(torch.cat(self.experiences[key][ids_in_batch]) for key in self.experiences.keys)

class PPOC:

    def __init__(self,
                 env,
                 n_options,
                 max_timesteps,
                 n_rollouts,
                 batch_size,
                 clip,
                 lr,
                 K) -> None:
        
        self.n_actions = env.action_space.shape[0]
        self.observation_dim = env.observation_space.shape[0]
        self.n_options = n_options
        self.env = env
        self.max_timesteps = max_timesteps
        self.n_rollouts = n_rollouts
        self.clip = clip
        self.lr = lr
        self.K = K
        self.prev_option = tensor(np.zeros(1)).long()
        self.is_init_state = tensor(np.ones(1)).byte()

        self.rollout_buffer = RolloutBuffer(batch_size=batch_size, buffer_size=self.n_rollouts)
        self.option_critic = OptionCritic(
            observation_dim=self.observation_dim,
            n_actions=self.n_actions,
            n_options=self.n_options
        )
        self.optimizer = optim.Adam(
            self.option_critic.parameters(),
            lr=self.lr,
            eps=1e-5)
    
    def compute_pi_h(self, next_option, beta):

        mask = torch.zeros_like(next_option)
        mask[self.previous_option] = 1
        is_init_state = self.is_init_state.view(-1, 1).expand(-1, next_option.size(1))
        pi_h = torch.where(self.is_init_state, next_option, beta * next_option + (1 - beta) * mask)
        return pi_h

    def _rollout_collections(self):
        obs, info = self.env.reset()
        cum_reward = 0
        with torch.no_grad(): 
            for step in range(self.n_rollouts):
                next_option, option_values, action_means, action_stds, betas = self.option_critic(obs)
                
                # determine next option
                pi_h = self.compute_pi_h(next_option, beta, is_init_states)
                option = torch.distributions.Categorical(probs = pi_h).sample()
                # determine next action based on option
                dist = torch.distributions.Normal(action_means[option], action_stds[option])
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)

                next_state, reward, done, truncated, info = self.env.step(action.cpu().detach().numpy())
                cum_reward += reward
                if done:
                    print("Cumulative reward at step " + str(step) +
                            " is " + str(cum_reward))
                    self.fd_train.write("%d %f\n" % (step, cum_reward))
                    self.fd_train.flush()
                    cum_reward = 0
                experience = {
                    "states": obs,
                    "actions": action,
                    "rewards": reward,
                    "log_probs": log_prob,
                    "option_values": option_value,
                    "betas": betas,
                    "options": option,
                    "previous_options": self.prev_option,
                    "is_init": self.is_init_state,
                    "masks": torch.FloatTensor(1 - done)
                }
                self.rollout_buffer.store(experience)
                self.is_init_state = tensor(done).byte()
                self.prev_option = option
                obs = new_state

    def _compute_advantages(self, rewards, dones, values):
        options, states, dones, option_values, previous_options = self.rollout_buffer.get(["options","states", "is_init_state", "option_values", "previous_options"])
        last_state, last_done = states[-1], dones[-1]
        beta_advs = []

        with torch.no_grad():
            option, option_value, action_means, action_stds, betas = self.option_critic(last_state)
            betas = betas[self.previous_option]
            ret = (1 - betass) * option_value[self.previous_option] + \
                betass * torch.max(option_value, dim=-1)[0]
            ret = ret.unsqueeze(-1)

        for i in reversed(range(self.roll_outs)):
#                v = qos[i].max(dim=-1, keepdim=True)[0] * (1 - self.eps) + qos[i].mean(-1).unsqueeze(-1) * self.eps
            v = (option_value * option).sum(-1).unsqueeze(-1)
            q = option_values[i].gather(1, previous_options[i])
            beta_advs.append(q - v + 0.01)
        values = option_values[range(options.shape[0]), options]
        rets = compute_gae(ret, rewards, masks, values)
        advs = rets - values
        advs = (advs-advs.mean())/advs.std()
        advantages = {
            "advantages" : advs,
            "beta_advantes": beta_advs
        }
        self.rollout_buffer.store(advantages)

    def _update(self):
        for _ in range(self.K):
        # actions, options, gae, old_log_probs
            for observations, actions, rewards, log_probs, advantages, beta_advantages, qos, betas, options, prev_options, entropies, masks in self.rollout_buffer:
                
                next_option, option_values, action_means, action_stds, betas = self.option_critic(observations)
                # means and stds -> [step, option, action], filter for chosen option
                action_mean_option = action_means[range(options.squeeze().shape[0]), options.squeeze()]
                action_std_option = action_stds[range(options.squeeze().shape[0]), options.squeeze()]
                # create normal distribution
                dist = torch.distributions.Normal(action_mean_option, action_std_option)

                #for each step, take log_prob of actual action, sum up log probabilities (product probs)
                new_log_probs = dist.log_prob(actions).sum(-1).unsqueeze(-1)
                
                # take the ratio between old and new
                ratio = (new_log_probs - old_log_probs).exp()
                # calculate ppoc loss and take mean over batch
                intra_option_policies_loss = -torch.min(
                    ratio * advantages, 
                    torch.clamp(ratio, 1.0 - self.clip,1.0 + self.clip) * advantages
                    ).mean()
                
                # calculate beta loss considering if it was stopped or not
                beta_loss = (betas.gather(1, prev_option) * beta_advantages (1 - masks)).mean()

                # MSE
                critic_loss = (option_values - rewards).pow(2).mean()

                A = -(next_option.gather(1, option) * adv).mean()
                B = (-(next_option*next_option.log()).sum(-1).mean())
                policy_over_options_loss =  A - 0.01 * B

                loss = 0.5 * critic_loss + intra_option_policies_loss  + beta_loss + policy_over_options_loss
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(option_critic.parameters(), 0.5)
                self.optimizer.step()

    def learn(self):
        path='./data/{}'.format(self.env_name)
        try:
            os.mkdir(path)
        except OSError as error:
            print(error)
        curtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S") \
                    + "_{:04d}".format(random.randint(1,9999))
        self.fd_train = open(path + '/ppoc_train_{}.log'.format(curtime), 'w')
        self._rollout_collections()
        self._compute_advantages()
        self._update()
        self.fd_train.close()


class OptionCritic(nn.Module):
    
    def __init__(self,
                 observation_dim,
                 n_actions,
                 n_options) -> None:
        super(OptionCritic, self).__init__()

        HIDDEN_DIMENSION = 64
        self.n_options = n_options

        self.policy_over_options = PolicyOverOptions(
            observation_dim, 
            n_options,
            HIDDEN_DIMENSION)

        self.q_option_net = OptionValueNet(observation_dim, n_options, HIDDEN_DIMENSION)
        self.intra_option_policies = nn.ModuleList(
            [IntraOptionPolicy(observation_dim, n_actions, HIDDEN_DIMENSION) for _ in range(n_options)]
        )

        self.termination_functions = nn.ModuleList(
            [TerminationFunction(observation_dim, HIDDEN_DIMENSION) for _ in range(n_options)]
        )

    def log_prob_of_action(self, x, action, option):
        return predict_action_dist(x, option).log_prob(action)
    
    def predict_action_dist(self, x, option):
        mean, std = self.intra_option_policies[option](x)
        dist = torch.distributions.Normal(mean, std)
        return dist

    def log_prob_of_option(self, x, option):
        next_option = self.policy_over_options(x)
        return next_option.log()

    def predict_termination(self, x, option):
        return self.termination_functions[option](x)

    def forward(self, x):
        
        option_values = self.q_option_net(x)
        next_option = self.policy_over_options(x)

        action_means = np.zeros(self.n_options)
        action_stds = np.zeros(self.n_options)
        betas = np.zeros(self.n_options)

        for i in range(self.n_options):
            mean, std = self.intra_option_policies[i](x)
            beta = self.termination_functions[i](x)
            action_means[i] = mean
            action_stds[i] = std
            betas[i] = beta
        
        action_means = torch.cat(action_means, dim=0)
        action_stds = torch.cat(action_stds, dim=0)
        betas = torch.cat(betas, dim=0)
        return next_option, option_values, action_means, action_stds, betas



class PolicyOverOptions(nn.Module):

    def __init__(self,
                 observation_dim,
                 n_options,
                 hidden_dim) -> None:
        super(PolicyOverOptions, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_options),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.net(x)
    
class OptionValueNet(nn.Module):

    def __init__(self, observation_dim, n_options, hidden_dim) -> None:
        super(OptionValueNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_options)
        )
    
    def forward(self, x):
        return self.net(x)
    
class IntraOptionPolicy(nn.Module):

    def __init__(self, observation_dim, action_dim, hidden_dim) -> None:
        super(IntraOptionPolicy, self).__init__()

        self.net_mean = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

        self.std = nn.Parameter(torch.zeros((1, action_dim)))

    def forward(self, x):
        mean = self.net_mean(x)
        std = F.softplus(self.std).expand(mean.size(0), -1)
        return mean, std
    
class TerminationFunction(nn.Module):
    
    def __init__(self, observation_dim, hidden_dim) -> None:
        super(TerminationFunction, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)





