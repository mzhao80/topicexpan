import torch
import torch.nn.functional as F


def infonce_loss(output, target, temperature=0.1):
    """
    InfoNCE loss for similarity prediction
    output: a (batch_size, num_classes) tensor
    target: a (batch_size, ) tensor of dtype long
    """
    # Apply temperature scaling
    output = output / temperature
    
    # Convert to log probabilities
    log_probs = F.log_softmax(output, dim=-1)
    
    # Get the loss for the correct class
    target_loss = torch.gather(log_probs, dim=-1, index=target.unsqueeze(-1))
    
    return -target_loss.mean()

def nll_loss(output, target):
    """
    Cross entropy loss for generation
    output: a (batch_size * sequence_length, vocab_size) tensor or (batch_size, sequence_length, vocab_size)
    target: a (batch_size * sequence_length) tensor or (batch_size, sequence_length)
    """
    # Handle both flattened and unflattened inputs
    if output.dim() == 3:
        batch_size, sequence_length, vocab_size = output.shape
        output = output.reshape(-1, vocab_size)
        target = target.reshape(-1)
    
    # Create mask for non-padding tokens
    non_pad_mask = (target != 0)
    
    # Apply log_softmax if not already applied
    if output.dim() == 2:
        output = F.log_softmax(output, dim=-1)
    
    # Calculate loss only on non-padding tokens
    loss = F.nll_loss(output, target, reduction='none')
    loss = loss * non_pad_mask.float()
    
    # Average over non-padding tokens
    num_tokens = non_pad_mask.sum().item()
    if num_tokens > 0:
        return loss.sum() / num_tokens
    return loss.sum()  # Return 0 if all padding
