#!/bin/bash
#SBATCH --partition=gpu_requeue      # Use the gpu_requeue partition
#SBATCH --gres=gpu:8                # Request 4 GPUs
#SBATCH --cpus-per-task=8           # Request 8 CPUs
#SBATCH --mem=20G                   # Request 20 GB of memory
#SBATCH --time=0-12:00:00           # Set the maximum runtime (12 hr)
#SBATCH --open-mode=append
#SBATCH --output=logs/train_%j.out       # Standard output log file (with job ID)
#SBATCH --error=logs/train_%j.err        # Standard error log file (with job ID)
#SBATCH --job-name=topicexpan_train # Job name

cd ~/Downloads/topicexpan
source myenv/bin/activate
module load cuda/11.8.0-fasrc01

python preprocess.py
