import subprocess
from tqdm import tqdm
import itertools

seeds = range(5)  # 0 to 4
js = [1, 2, 4, 8, 16]
env_name = "Swimmer-v2"  # Change this if needed

# Generate all combinations of seed and j
combos = list(itertools.product(seeds, js))

# Loop with progress bar
for seed, j in tqdm(combos, desc="Evaluating SAC", total=len(combos)):
    cmd = [
        "python", "eval_sac.py",
        "--env_name", env_name,
        "--seed", str(seed),
        "--j", str(j)
    ]
    subprocess.run(cmd)
