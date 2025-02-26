#!/bin/bash
#SBATCH --partition=gpu_requeue      # Use the gpu_requeue partition
#SBATCH --gres=gpu:8                # Request 4 GPUs
#SBATCH --cpus-per-task=8           # Request 8 CPUs
#SBATCH --mem=20G                   # Request 20 GB of memory
#SBATCH --time=0-12:00:00           # Set the maximum runtime (12 hr)
#SBATCH --open-mode=append
#SBATCH --output=logs/train_full_%j.out       # Standard output log file (with job ID)
#SBATCH --error=logs/train_full_%j.err        # Standard error log file (with job ID)
#SBATCH --job-name=topicexpan_full # Job name

echo "Job started at $(date)"

cd ~/Downloads/topicexpan
echo "Activating virtual environment at $(date)"
source myenv/bin/activate
module load cuda/11.8.0-fasrc01

echo "Starting preprocessing at $(date)"
python preprocess.py --data-dir congress_full

echo "Making topic triples human-readable at $(date)"
python analyze_data.py --data-dir congress_full

echo "Generating dataset binary at $(date)"
python generate_dataset_binary.py --data_dir congress_full

echo "Creating save directories at $(date)"
mkdir -p congress_full-save/models congress_full-save/log

# Find the most recent checkpoint
CHECKPOINT_DIR="congress_full-save/models"
LATEST_CHECKPOINT=$(ls -t $CHECKPOINT_DIR/checkpoint-epoch*.pth | head -n 1)

# Check if a checkpoint was found
echo "Checking for checkpoint at $(date)"
if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "No checkpoint found. Starting training from scratch."
    echo "Starting training at $(date)"
    python train.py --config config_files/config_congress_full.json
else
    echo "Resuming from checkpoint: $LATEST_CHECKPOINT"
    echo "Starting training at $(date)"
    python train.py --config config_files/config_congress_full.json --resume "$LATEST_CHECKPOINT"
fi

echo "Starting expansion at $(date)"
python expand.py --config config_files/config_congress_full.json --resume congress_full-save/models/model_best.pth

echo "Job completed at $(date)"
