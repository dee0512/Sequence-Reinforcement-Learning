hyperparametersSAC = {
    'Default': {
        'alpha':0.2,
        'hidden_size':256,
        'max_timesteps': 1000000,
        'eval_freq': 5000,
        'start_timesteps': 10000,
        'updates_per_step':1,
        'target_update_interval':1,
        'discount': 0.99,
        'tau': 0.005,
        'lr':0.0003,
        'noise_clip': 0.5,
        'policy_freq': 2,
        'replay_size': 1000000,
        'expl_noise': 0.1,
        'batch_size': 256,
        'max_episode_steps': 1000
    },
    'Hopper-v2': {
    },
    'Walker2d-v2': {
    },
    'Ant-v2': {

        'max_timesteps': 5000000,
    },
    'HalfCheetah-v2': {
            'max_timesteps': 5000000,
    },
    'Humanoid-v2':{
        'max_timesteps': 10000000
    },
    'HumanoidStandup-v2':{
            'max_timesteps': 1000000
        },
    'Pendulum-v1': {
        'eval_freq': 2500,
        'max_timesteps': 30000,
        'max_episode_steps':200,
        'start_timesteps': 1000,
    },
    'LunarLanderContinuous-v2': {
        'start_timesteps': 10000,
        'eval_freq': 2500,
        'max_timesteps': 500000,
    },
    'InvertedPendulum-v2': {
    },
    'InvertedDoublePendulum-v2': {
    },
    'Reacher-v2':{
        'max_timesteps': 500000
    },
    'Swimmer-v2':{
    }
}


def get_hyperparameters(env_name, type="SAC"):
    if type == "SAC":
        obj = hyperparametersSAC['Default']
        for key in hyperparametersSAC[env_name].keys():
            obj[key] = hyperparametersSAC[env_name][key]
        return obj