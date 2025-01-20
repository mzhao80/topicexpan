import torch
import torch.nn.functional as F


def infonce_loss(output, target, temperature=0.07):
    """
    output: a (batch_size, num_classes) tensor of similarity scores
    target: a (batch_size,) tensor of correct class indices
    """
    print(f"[DEBUG] InfoNCE input shapes - output: {output.shape}, target: {target.shape}")
    print(f"[DEBUG] Raw similarity scores min/max/mean: {output.min().item():.3f}/{output.max().item():.3f}/{output.mean().item():.3f}")
    
    # Get target scores before any scaling
    target_scores = output[torch.arange(output.size(0)), target]
    print(f"[DEBUG] Target scores before scaling min/max/mean: {target_scores.min().item():.3f}/{target_scores.max().item():.3f}/{target_scores.mean().item():.3f}")
    
    # Apply temperature scaling
    output = output / temperature
    print(f"[DEBUG] After temperature scaling min/max/mean: {output.min().item():.3f}/{output.max().item():.3f}/{output.mean().item():.3f}")
    
    # Get target scores after scaling
    target_scores = output[torch.arange(output.size(0)), target]
    print(f"[DEBUG] Target scores after scaling min/max/mean: {target_scores.min().item():.3f}/{target_scores.max().item():.3f}/{target_scores.mean().item():.3f}")
    
    # Compute log_softmax
    log_probs = F.log_softmax(output, dim=-1)
    print(f"[DEBUG] After log_softmax min/max/mean: {log_probs.min().item():.3f}/{log_probs.max().item():.3f}/{log_probs.mean().item():.3f}")
    
    # Compute target log probabilities
    target_log_probs = log_probs[torch.arange(output.size(0)), target]
    print(f"[DEBUG] Target log probs min/max/mean: {target_log_probs.min().item():.3f}/{target_log_probs.max().item():.3f}/{target_log_probs.mean().item():.3f}")
    
    # Compute loss
    loss = -target_log_probs.mean()
    print(f"[DEBUG] Loss value: {loss.item():.3f}")
    
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
