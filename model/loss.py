import torch
import torch.nn.functional as F


def infonce_loss(output, target, temperature=0.07, l2_reg=0.01):
    """
    output: a (batch_size, num_classes) tensor
    target: a (batch_size, ) tensor of dtype long
    temperature: temperature for scaling (lower = sharper distribution)
    l2_reg: L2 regularization strength
    """
    # L2 regularization
    l2_loss = l2_reg * (output ** 2).mean()
    
    # InfoNCE loss
    output = torch.softmax(output/temperature, dim=0)
    output = torch.gather(output, dim=1, index=target[:, None])
    loss = -torch.log(output + 1e-12)
    
    return loss.sum() + l2_loss

def nll_loss(output, target, label_smoothing=0.1):
    """
    output: a (batch_size, sequence_length, vocab_size) tensor
    target: a (batch_size, sequence_length) tensor
    label_smoothing: amount of label smoothing
    """
    batch_size, sequence_length, vocab_size = output.shape
    output = output.reshape(-1, vocab_size)
    target = target.reshape(-1)
    
    # Apply label smoothing
    return F.cross_entropy(
        output, target,
        ignore_index=0,
        reduction="sum",
        label_smoothing=label_smoothing
    )
