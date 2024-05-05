import gymnasium as gym
import os
import random
import string
import argparse

import torch

from obstacle_env import QuadXObstacleEnv
from PyFlyt.gym_envs import FlattenWaypointEnv

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO

def make_env(environment_id, log_dir):
    def _thunk():
        if environment_id=="QuadX-Hover-v1":
            env = gym.make(f"PyFlyt/{environment_id}")
        elif environment_id=="QuadX-Wapoints-1":
            env = gym.make(f"PyFlyt/{environment_id}")
            env = FlattenWaypointEnv(env, context_lenght=1)
        elif environment_id=="QuadX-Obstacles-1":
            env = QuadXObstacleEnv()
        else:
            raise "Uncompatible environment"
        env = gym.wrappers.NormalizeObservation(env)
        env = Monitor(env, log_dir)
        return env
    return _thunk

def train(args):
    environment = args['environment']
    id = ''.join(random.choices(string.ascii_letters, k=20))
    full_id = "PPO_" + '_' + environment + '_' + id

    models_dir = f"models/{full_id}/"
    logdir = models_dir + "data"
    checkpoint_dir = models_dir + "checkpoints/"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    
    train_env = make_vec_env(
        make_env(
            environment_id=environment, 
            log_dir=logdir if args["log"]else None), 
        n_envs=1)

    checkpoint_callback = CheckpointCallback(
        save_freq=1000000,
        save_path=checkpoint_dir,
        name_prefix="PPO_" + environment
    )

    if args["sde"]:
        policy_kwargs = dict(
            log_std_init=-2,
            ortho_init=False,
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[256,256], vf=[256,256])
        )
        model = PPO(
            policy="MlpPolicy", 
            env=train_env,
            batch_size=128,
            n_steps=512,
            gamma=0.99,
            gae_lambda=0.9,
            ent_coef=0.0,
            sde_sample_freq=4,
            max_grad_norm=0.5,
            vf_coef=0.5,
            learning_rate=3e-5,
            use_sde=True,
            clip_range=0.4,
            policy_kwargs=policy_kwargs, 
            verbose=1, 
            tensorboard_log=logdir if args["log"] else None)
    else:
        model = PPO(
            policy="MlpPolicy", 
            env=train_env, 
            verbose=1, 
            tensorboard_log=logdir if args["log"] else None)

    try:
        model.learn(total_timesteps=args["total_timesteps"],
                    callback=checkpoint_callback,
                    tb_log_name="tensorboard")
    except KeyboardInterrupt:
        model_name = f"{checkpoint_dir}/{model.num_timesteps}"
        model.save(model_name)
        with open('recent_model.txt', 'w') as file:
            file.write(model_name)
     
    train_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an agent in an environment using a DRL algorithm from stable baselines 3')

    parser.add_argument('--sde', type=bool, default=False, help='uses sde')
    parser.add_argument('--log', type=bool, default=True, help='record logs to tensorboard')
    parser.add_argument('--environment', '-env', type=str, default="QuadX-Hover-v1", help='which environment to train on')
    parser.add_argument('--total_timesteps', '-tim', type=int, default=2000000, help='training duration')
    args = parser.parse_args()

    train(vars(args))