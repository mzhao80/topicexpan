import torch
import torch.nn.functional as F


def infonce_loss(output, target, temperature=0.1):
    """
    output: a (batch_size, num_classes) tensor of similarity scores
    target: a (batch_size,) tensor of correct class indices
    """
    print(f"[DEBUG] InfoNCE input shapes - output: {output.shape}, target: {target.shape}")
    print(f"[DEBUG] Target values: {target}")
    
    output = output / temperature
    output = F.log_softmax(output, dim=-1)  # Use log_softmax for numerical stability
    print(f"[DEBUG] After log_softmax shape: {output.shape}")
    
    # Gather the scores for the target topics
    loss = F.nll_loss(output, target)
    print(f"[DEBUG] Loss value: {loss.item()}")
    
    return loss

def nll_loss(output, target):
    """
    output: a (batch_size, sequence_length, vocab_size) tensor
    target: a (batch_size, sequence_length) tensor
    """
    batch_size, sequence_length, vocab_size = output.shape
    output = output.reshape(-1, vocab_size)
    target = target.reshape(-1)
    return F.nll_loss(output, target, ignore_index=0, reduction="sum")
