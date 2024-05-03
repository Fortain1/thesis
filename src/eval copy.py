import gymnasium
import os
import argparse

from PyFlyt.gym_envs import FlattenWaypointEnv
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC

def eval(args):

    environment = args['environment']

    print(args['model'])

    algorithm_name = args['model'][args['model'].find('/')+1:args['model'].find('_')]
    if algorithm_name == "PPO": algorithm = PPO
    elif algorithm_name == "A2C": algorithm = A2C
    elif algorithm_name == "DDPG": algorithm = DDPG
    elif algorithm_name == "TD3": algorithm = TD3
    elif algorithm_name == "SAC": algorithm = SAC
    else:
        print("Error: Invalid DRL Algorithm specified")
        return

    model_path = f"{args['model']}"

    env = make_vec_env(lambda: FlattenWaypointEnv(gymnasium.make(f"PyFlyt/{environment}"), context_length=1), n_envs=1)
    model = algorithm.load(model_path, env=env)

    for _ in range(args['eval_episodes']):
        render_env = FlattenWaypointEnv(gymnasium.make(f"PyFlyt/{environment}", render_mode="human"), context_length=1)
        obs = render_env.reset()
        obs = obs[0]
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, terminated, truncated, info = render_env.step(action)
            done = terminated or truncated
            render_env.render()
        render_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a trained agent in an environment')

    with open('recent_model.txt', 'r') as file:
        recent_model = file.read().strip()

    parser.add_argument('--model', '-m', type=str, default=recent_model, help='which saved model to evaluate')
    parser.add_argument('--environment', '-env', type=str, default="QuadX-Waypoints-v1", help='which environment to train on')
    parser.add_argument('--eval_episodes', '-ee', type=int, default=10, help='the number of episodes to evaluate on')
    args = parser.parse_args()

    eval(vars(args))