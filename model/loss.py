import torch
import torch.nn.functional as F


def infonce_loss(output, target, temperature=0.1):
    """
    output: a (batch_size, num_classes) tensor
    target: a (batch_size, ) tensor of dtype long
    """
    print(f"[DEBUG] InfoNCE input shapes - output: {output.shape}, target: {target.shape}")
    print(f"[DEBUG] InfoNCE input sample - output[0]: {output[0]}, target[0]: {target[0]}")
    
    if output.dim() == 1:
        # Handle case where output is 1D
        output = output.unsqueeze(0)  # Add batch dimension
        print(f"[DEBUG] Reshaped 1D output to: {output.shape}")
    
    output = torch.softmax(output/temperature, dim=-1)
    print(f"[DEBUG] After softmax shape: {output.shape}")
    
    # Use gather for safer indexing
    index = target.view(-1, 1)
    output = torch.gather(output, dim=-1, index=index).squeeze(-1)
    print(f"[DEBUG] After gather shape: {output.shape}")
    
    loss = - torch.log(output + 1e-12)
    print(f"[DEBUG] Loss shape: {loss.shape}, mean: {loss.mean().item()}")
    return loss.mean()

def nll_loss(output, target):
    """
    output: a (batch_size, sequence_length, vocab_size) tensor
    target: a (batch_size, sequence_length) tensor
    """
    batch_size, sequence_length, vocab_size = output.shape
    output = output.reshape(-1, vocab_size)
    target = target.reshape(-1)
    return F.nll_loss(output, target, ignore_index=0, reduction="sum")
