#!/bin/bash

#SBATCH -J windowII+l2+slow_subset            # Job name (optional)
#SBATCH -p scc-gpu                  # Partition (GPU queue)
#SBATCH -c 32                       # Number of CPU cores
#SBATCH -t 45:00:00                 # Wall time (hh:mm:ss)
#SBATCH --mem=128G                # System RAM (main memory)
#SBATCH -G A100:4          # Request  NVIDIA A100 GPU
#SBATCH -N 1
#SBATCH -o logs/windowII+l2+slow_subset.out   # Standard output
#SBATCH -e logs/windowII+l2+slow_subset.err   # Standard erro


# 1. Set up micromamba
eval "$(micromamba shell hook --shell=bash)"

# 2. Activate environment
micromamba activate env

# 3. (Optional but helpful) Show which Python is active
which python
python --version

export CUDA_LAUNCH_BLOCKING=1

# 4. Run your script
python -u windowII+l2+slow_subset.py
