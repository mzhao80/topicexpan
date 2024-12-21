#!/bin/bash
#SBATCH --partition=gpu_requeue      # Use the gpu_requeue partition
#SBATCH --gres=gpu:2                # Request 2 GPUs
#SBATCH --cpus-per-task=4           # Request 4 CPUs
#SBATCH --mem=10G                   # Request 10 GB of memory
#SBATCH --time=1-00:00:00           # Set the maximum runtime (1 day)
#SBATCH --open-mode=append
#SBATCH --output=logs/train_%j.out       # Standard output log file (with job ID)
#SBATCH --error=logs/train_%j.err        # Standard error log file (with job ID)
#SBATCH --job-name=topicexpan_train # Job name

cd ~/Downloads/topicexpan
source myenv/bin/activate

# Find the most recent checkpoint
CHECKPOINT_DIR="amazon-save/models"
LATEST_CHECKPOINT=$(ls -t $CHECKPOINT_DIR/checkpoint-epoch*.pth | head -n 1)

# Check if a checkpoint was found
if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "No checkpoint found. Starting training from scratch."
    python train.py --config config_files/config_amazon.json
else
    echo "Resuming from checkpoint: $LATEST_CHECKPOINT"
    python train.py --config config_files/config_amazon.json --resume "$LATEST_CHECKPOINT"
fi
