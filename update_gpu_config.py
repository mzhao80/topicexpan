import json
import torch
import os

def update_gpu_config(config_path):
    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    
    # Get list of available GPU devices
    for i in range(num_gpus):
        gpu_info = torch.cuda.get_device_properties(i)
        print(f"Found GPU {i}: {gpu_info.name} with {gpu_info.total_memory / 1024**3:.1f}GB memory")
    
    print(f"\nTotal GPUs available: {num_gpus}")
    
    # Read existing config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Update GPU configuration - remove device since we'll use all GPUs
    if 'device' in config:
        del config['device']
    if 'n_gpu' in config:
        del config['n_gpu']
    
    # Backup original config
    backup_path = config_path + '.backup'
    if not os.path.exists(backup_path):
        with open(backup_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"\nBackup created at: {backup_path}")
    
    # Write updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"\nUpdated {config_path} - removed GPU configuration to use all available GPUs")

if __name__ == "__main__":
    config_path = os.path.join('config_files', 'config_congress.json')
    update_gpu_config(config_path)
