# Sequence Reinforcement Learning

## 🚀 Getting Started

### Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 📈 Neptune Logging
This script automatically logs training metrics to Neptune.ai. Make sure your neptune_config.py file is set up as follows:

```
# neptune_config.py
NEPTUNE_PROJECT = "your-username/your-project"
NEPTUNE_API_TOKEN = "your-api-token"
```

## Directory Structure
All code is provided in the scripts folder

## Reproducing Results:

### SAC

#### Training
To reproduce the SAC experiments referenced in Table 2, Figures 2, 3, 5, 7, 8, 10, 11, 12, 13, from the scripts folder, run the train_sac.py file for training:

```commandline
python .\train_sac.py --env_name <env_name> --seed <seed> --j <j>
```

#### Evaluation:
```commandline
python .\eval_sac.py --env_name <env_name> --seed <seed> --j <j>
```
Alternatively you can also run the ```run_eval_grid.py```


### Arguments:

| Argument                     | Description                                                  |
| ---------------------------- | ------------------------------------------------------------ |
| `--env_name`                 | Name of the Gym environment (default: `InvertedPendulum-v2`) |
| `--seed`                     | Random seed for reproducibility                              |
| `--j`                        | Frame-skipping multiplier (default: `1`)                     |

### Supported Environments
```
Hopper-v2  
Walker2d-v2  
Ant-v2  
HalfCheetah-v2  
Humanoid-v2
Pendulum-v1  
LunarLanderContinuous-v2  
InvertedPendulum-v2  
InvertedDoublePendulum-v2  
Reacher-v2  
Swimmer-v2
```

### Trained Models
Get trained models on Hugging Face:
#### [SRL HuggingFace Collection](https://huggingface.co/collections/devdharpatel/sequence-reinforcement-learning-68845f6b7d657cbe7dab9282)
