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

mkdir -p congress-save/models congress-save/log

# Find the most recent checkpoint
CHECKPOINT_DIR="congress-save/models"
LATEST_CHECKPOINT=$(ls -t $CHECKPOINT_DIR/checkpoint-epoch*.pth | head -n 1)

# Check if a checkpoint was found
if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "No checkpoint found. Starting training from scratch."
    python train.py --config config_files/config_congress.json
else
    echo "Resuming from checkpoint: $LATEST_CHECKPOINT"
    python train.py --config config_files/config_congress.json --resume "$LATEST_CHECKPOINT"
fi
