import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
from openpyxl import Workbook

from utils import make_env, create_folders
from hyperparameters import get_hyperparameters
from sac import SAC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure the logs directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "evaluation.log")),
        logging.StreamHandler()
    ]
)


def create_excel_if_not_exists(file_path):
    """
    Create an Excel file if it does not already exist.

    Args:
        file_path (str): Path to the Excel file.
    """
    if not os.path.exists(file_path):
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Sheet"
        workbook.save(filename=file_path)
        logging.info(f"New workbook created and saved as {file_path}")
    else:
        logging.info(f"Workbook already exists at {file_path}")


def setup_environment(env_name, seed):
    """
    Set up the environment based on the type of environment.

    Args:
        env_name (str): Name of the environment.
        seed (int): Random seed for reproducibility.

    Returns:
        Environment object.
    """
    return make_env(env_name, seed)


def evaluate_policy(policy, eval_env, action_dim, steps=2):
    """
    Evaluate the policy and return the average reward.

    Args:
        policy: The policy to be evaluated.
        eval_env: The environment for evaluation.
        action_dim (int): Dimension of the action space.
        steps (int): Number of steps to plan ahead.

    Returns:
        float: Average reward over the evaluation episodes.
    """
    rewards = 0
    for _ in range(10):
        eval_state, eval_done = eval_env.reset(), False
        eval_episode_timesteps = 0
        eval_prev_action = torch.zeros(action_dim)
        while not eval_done:
            eval_action = policy.select_action(eval_state, evaluate=True)

            for eval_ps in range(steps):
                eval_next_state, eval_reward, eval_done, _ = eval_env.step(eval_action)
                eval_state = eval_next_state
                eval_episode_timesteps += 1
                rewards += eval_reward
                if eval_done:
                    break
    avg_reward = rewards / 10
    return avg_reward


def eval(seed=0, env_name='InvertedPendulum-v2', j=1):
    """
    Main function to evaluate the policy. Model is trained and evaluated inside.

    Args:
        seed (int): Random seed for reproducibility.
        env_name (str): Name of the environment.
        automatic_entropy_tuning (bool): Whether to automatically tune entropy.
        j (int): Frameskip multiplier
    """
    hy = get_hyperparameters(env_name, 'SAC')

    augment_type = "SAC"
    arguments = [augment_type, env_name, seed, j]
    file_name = '_'.join([str(x) for x in arguments])

    logging.info("---------------------------------------")
    logging.info(f"Env: {env_name}, Seed: {seed}")
    logging.info("---------------------------------------")

    create_folders()

    env = setup_environment(env_name, seed)

    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy_kwargs = {
        "gamma": hy['discount'],
        "tau": hy['tau'],
        "alpha": hy['alpha'],
        "policy_type": "Gaussian",
        "hidden_size": hy['hidden_size'],
        "target_update_interval": hy['target_update_interval'],
        "automatic_entropy_tuning": True,
        "lr": hy['lr'],
    }

    policy = SAC(state_dim, env.action_space, **policy_kwargs)
    policy.load_checkpoint(f"./models/{file_name}_best")

    eval_env = setup_environment(env_name, seed + 100)

    steps_list = [x for x in range(2, 32, 2)]
    steps_list.insert(0, 1)
    for s in steps_list:
        logging.info(f"Evaluating with steps: {s}")
        avg_reward = evaluate_policy(policy, eval_env, action_dim, steps=s)
        logging.info(f" --------------- Evaluation reward {avg_reward:.3f}")

        df1 = pd.DataFrame({
            'seed': [seed],
            'reward': [avg_reward],
            'env_name': [env_name],
            'j': [j],
            'steps': [s]
        })
        df1.to_csv('evalsac.csv', mode='a', index=False, header=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="InvertedPendulum-v2", help="Environment name")
    parser.add_argument("--seed", default=0, type=int, help="Sets Gym, PyTorch and Numpy seeds")
    parser.add_argument('--j', default=1, type=int, help="Frameskip multiplier")

    args = vars(parser.parse_args())
    logging.info('Command-line argument values:')
    for key, value in args.items():
        logging.info(f'- {key} : {value}')

    eval(**args)
