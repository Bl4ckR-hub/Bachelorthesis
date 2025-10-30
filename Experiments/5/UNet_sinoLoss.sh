#!/bin/bash

#SBATCH -J UNet_sino           # Job name (optional)
#SBATCH -p scc-gpu                  # Partition (GPU queue)
#SBATCH -c 32                       # Number of CPU cores
#SBATCH -t 45:00:00                 # Wall time (hh:mm:ss)
#SBATCH --mem=128G                # System RAM (main memory)
#SBATCH -G A100:4          # Request  NVIDIA A100 GPU
#SBATCH -N 1
#SBATCH -o logs/UNet_sino.out   # Standard output
#SBATCH -e logs/UNet_sino.err   # Standard error


# 1. Set up micromamba
eval "$(micromamba shell hook --shell=bash)"

# 2. Activate environment
micromamba activate env

# 3. (Optional but helpful) Show which Python is active
which python
python --version

export CUDA_LAUNCH_BLOCKING=1

# 4. Run your script
python -u UNet_sinoLoss.py
