import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel

def mean_pooling(model_output, attention_mask):
    # Mean Pooling - Take attention mask into account for correct averaging
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class EmbeddingSimilarity:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.__name__ = 'embedding_sim'

    def __call__(self, output_ids, target_ids, attention_mask=None, target_attention_mask=None):
        # Compute token embeddings directly from IDs
        with torch.no_grad():
            if attention_mask is None:
                attention_mask = (output_ids != 0).float()
            if target_attention_mask is None:
                target_attention_mask = (target_ids != 0).float()
                
            output_embeds = self.model(input_ids=output_ids, attention_mask=attention_mask)
            target_embeds = self.model(input_ids=target_ids, attention_mask=target_attention_mask)

        # Perform pooling
        output_embeds = mean_pooling(output_embeds, attention_mask)
        target_embeds = mean_pooling(target_embeds, target_attention_mask)

        # Normalize embeddings
        output_embeds = F.normalize(output_embeds, p=2, dim=1)
        target_embeds = F.normalize(target_embeds, p=2, dim=1)

        # Compute cosine similarity
        similarities = F.cosine_similarity(output_embeds, target_embeds)
        return similarities.mean().item()

def perplexity(output, target):
    batch_size, sequence_length, vocab_size = output.shape
    output = output.reshape(-1, vocab_size)
    target = target.reshape(-1)
    pp = F.nll_loss(output, target, ignore_index=0, reduction="sum")
    return pp / batch_size

def accuracy(output, target):
    correct = 0
    for output_str, target_str in zip(output, target):
        if output_str == target_str:
            correct += 1
    return correct / len(target)

# Create module-level instance
embedding_sim = EmbeddingSimilarity()