import argparse
import logging
import os
import numpy as np
import torch
import neptune

from utils import make_env, create_folders
from replay_buffers import ReplayBuffer
from hyperparameters import get_hyperparameters
from sac import SAC
from neptune_config import NEPTUNE_PROJECT, NEPTUNE_API_TOKEN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Logging setup ---
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "training.log")),
        logging.StreamHandler()
    ]
)


def setup_neptune(parameters):
    run = neptune.init_run(
        project=NEPTUNE_PROJECT,
        api_token=NEPTUNE_API_TOKEN,
    )
    run["parameters"] = parameters
    return run


def setup_environment(env_name, seed):
    return make_env(env_name, seed)


def evaluate_policy(policy, env_name, seed, j):
    eval_env = make_env(env_name, seed + 100)
    rewards = 0
    for _ in range(10):
        eval_state, eval_done = eval_env.reset(), False
        while not eval_done:
            eval_action = policy.select_action(eval_state, evaluate=True)
            for _ in range(j):
                eval_next_state, eval_reward, eval_done, _ = eval_env.step(eval_action)
                rewards += eval_reward
                if eval_done:
                    break
            eval_state = eval_next_state
    return rewards / 10


def train(seed=0, env_name='InvertedPendulum-v2', j=1):

    parameters = {
        'type': "SAC",
        'env_name': env_name,
        'seed': seed,
        'j': j
    }
    run = setup_neptune(parameters)

    hy = get_hyperparameters(env_name, 'SAC')
    file_name = f"SAC_{env_name}_seed{seed}_j{j}"

    logging.info(f"Environment: {env_name} | Seed: {seed}")
    create_folders()

    env = setup_environment(env_name, seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    max_timesteps = hy['max_timesteps']
    eval_freq = hy['eval_freq']
    start_timesteps = hy['start_timesteps']
    max_episode_timestep = hy['max_episode_steps']

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = SAC(state_dim, env.action_space, **{
        "gamma": hy['discount'],
        "tau": hy['tau'],
        "alpha": hy['alpha'],
        "policy_type": "Gaussian",
        "hidden_size": hy['hidden_size'],
        "target_update_interval": hy['target_update_interval'],
        "automatic_entropy_tuning": True,
        "lr": hy['lr'],
    })

    replay_buffer = ReplayBuffer(state_dim, action_dim)
    state, done = env.reset(), False
    episode_reward, episode_timesteps, episode_num, updates = 0, 0, 0, 0
    best_performance = -np.inf
    evaluations = []

    t = 0
    while t < max_timesteps:
        action = env.action_space.sample() if t < start_timesteps else policy.select_action(state)

        total_reward = 0
        for _ in range(j):
            next_state, reward, done, _ = env.step(action)
            episode_timesteps += 1
            done_bool = float(done) if episode_timesteps < max_episode_timestep else 0
            episode_reward += reward
            total_reward += reward
            t += 1

            if (t + 1) % eval_freq == 0 and t >= start_timesteps:
                avg_reward = evaluate_policy(policy, env_name, seed, j)
                evaluations.append(avg_reward)
                logging.info(f"Evaluation @ step {t + 1}: avg_reward = {avg_reward:.3f}")
                run['avg_reward'].log(avg_reward)
                np.save(f"./results/{file_name}", evaluations)

                if avg_reward > best_performance:
                    best_performance = avg_reward
                    run['best_reward'].log(best_performance)
                    policy.save_checkpoint(f"./models/{file_name}_best")

            if done:
                break

        replay_buffer.add(state, action, next_state, total_reward, done_bool)
        state = next_state

        if replay_buffer.size >= hy["batch_size"]:
            losses = policy.update_parameters(replay_buffer, hy["batch_size"], updates)
            critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = losses
            updates += 1

            run['critic1'].log(critic_1_loss)
            run['critic2'].log(critic_2_loss)
            run['policy'].log(policy_loss)
            run['entropy'].log(ent_loss)
            run['alpha'].log(alpha)

        if done:
            logging.info(
                f"Total T: {t} | Episode: {episode_num + 1} | Ep T: {episode_timesteps} | Reward: {episode_reward:.2f}")
            state, done = env.reset(), False
            episode_reward, episode_timesteps = 0, 0
            episode_num += 1

    policy.save_checkpoint(f"./models/{file_name}_final")
    run.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="InvertedPendulum-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--j", type=int, default=1)

    args = parser.parse_args()
    logging.info("Launching training with arguments:")
    for arg, val in vars(args).items():
        logging.info(f"  {arg} = {val}")

    train(**vars(args))
