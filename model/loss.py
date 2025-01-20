import torch
import torch.nn.functional as F


def infonce_loss(output, target, temperature=0.1):
    """
    output: a (batch_size, num_classes) tensor of similarity scores
    target: a (batch_size,) tensor of target class indices
    """
    # Apply temperature scaling and softmax
    output = output / temperature
    log_probs = F.log_softmax(output, dim=-1)
    
    # Get the predicted probability for the target class
    target_log_probs = log_probs[torch.arange(output.size(0)), target]
    
    # InfoNCE loss is the negative log probability of the target class
    loss = -target_log_probs.mean()
    
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
