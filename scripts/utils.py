import gym
import os
__all__ = ["make_env", "create_folders"]


def make_env(env_name, seed):
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)

    return env


def create_folders():
    if not os.path.exists("./results"):
        os.makedirs("./results")

    if not os.path.exists("./models"):
        os.makedirs("./models")