#!/bin/bash
#SBATCH --partition=gpu_requeue      # Use the gpu_requeue partition
#SBATCH --gres=gpu:2                # Request 2 GPUs
#SBATCH --cpus-per-task=4           # Request 4 CPUs
#SBATCH --mem=10G                   # Request 10 GB of memory
#SBATCH --time=1-00:00:00           # Set the maximum runtime (1 day)
#SBATCH --open-mode=append
#SBATCH --output=logs/expand_%j.out       # Standard output log file (with job ID)
#SBATCH --error=logs/expand_%j.err        # Standard error log file (with job ID)
#SBATCH --job-name=topicexpan_expand # Job name

cd ~/Downloads/topicexpan
source myenv/bin/activate

python expand.py --config config_files/config_amazon.json --resume amazon-save/models/model_best.pth
