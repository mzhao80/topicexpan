import json
import torch
import os

def update_gpu_config(config_path):
    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    
    # Get list of available GPU devices
    gpu_devices = []
    for i in range(num_gpus):
        gpu_info = torch.cuda.get_device_properties(i)
        print(f"Found GPU {i}: {gpu_info.name} with {gpu_info.total_memory / 1024**3:.1f}GB memory")
        gpu_devices.append(i)
    
    print(f"\nTotal GPUs available: {num_gpus}")
    
    # Read existing config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Update GPU configuration
    config['n_gpu'] = num_gpus
    config['device'] = gpu_devices if gpu_devices else [-1]  # Use [-1] for CPU
    
    # Backup original config
    backup_path = config_path + '.backup'
    if not os.path.exists(backup_path):
        with open(backup_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"\nBackup created at: {backup_path}")
    
    # Write updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"\nUpdated {config_path} with:")
    print(f"n_gpu: {config['n_gpu']}")
    print(f"device: {config['device']}")

if __name__ == "__main__":
    config_path = os.path.join('config_files', 'config_congress.json')
    update_gpu_config(config_path)
