import math
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init, TransformerDecoderLayer, TransformerDecoder

import dgl
import dgl.function as fn
from dgl.nn.pytorch.conv import GraphConv
from transformers import AutoModel, AutoConfig
from base import BaseModel

"""
    1. Document Encoder
"""
class BertDocEncoder(BaseModel):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        self.input_embeddings = self.model.embeddings
        
    def forward(self, x):
        """
        x : a dict of bert required import
        return: a tensor of shape (batch_size, doc_embed_dim)
        """
        batch_output = self.model(**x)
        return batch_output[0]

"""
    2. Topic Encoder
"""
class GCNTopicEncoder(BaseModel):
    def __init__(self, topic_hier, topic_node_feats, topic_mask_feats, topic_num_layers): 
        super(GCNTopicEncoder, self).__init__()
        num_topics, topic_embed_dim = topic_node_feats.shape
        in_dim, hidden_dim, out_dim = topic_embed_dim, topic_embed_dim, topic_embed_dim

        self.num_layers = topic_num_layers
        self.activation = F.leaky_relu
        
        # Convert to tensors and move to GPU if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.topic_node_feats = torch.tensor(topic_node_feats, device=device)
        self.topic_mask_feats = torch.tensor(topic_mask_feats, device=device)
        self.topic_hier, self.num_topics = topic_hier, num_topics
        
        # Generate and move adjacency matrices to GPU
        self.downward_adjmat, self.upward_adjmat, self.sideward_adjmat = self._generate_adjmat(topic_hier, num_topics)
        self.downward_adjmat = self.downward_adjmat.to(device)
        self.upward_adjmat = self.upward_adjmat.to(device)
        self.sideward_adjmat = self.sideward_adjmat.to(device)

        self.downward_layers, self.upward_layers, self.sideward_layers = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()

        for layers in [self.downward_layers, self.upward_layers, self.sideward_layers]:
            layers.append(GraphConv(in_dim, hidden_dim, norm='right', allow_zero_in_degree=True))
            for l in range(self.num_layers - 2):
                layers.append(GraphConv(hidden_dim, hidden_dim, norm='right', allow_zero_in_degree=True))
            layers.append(GraphConv(hidden_dim, out_dim, norm='right', allow_zero_in_degree=True))
        
        # Move model to GPU
        self.to(device)

    def _generate_adjmat(self, topic_hier, num_topics, virtual_src=None, virtual_dst=None):
        vsrc, vdst, hsrc, hdst = [], [], [], []
        for parent, childs in topic_hier.items():
            vsrc += [parent] * len(childs) 
            vdst += [child for child in childs]
            for src, dst in itertools.permutations(childs, 2):
                hsrc += [src]
                hdst += [dst]

        # Add a virtual node and its corresponding edges
        if virtual_src is not None and virtual_dst is not None:
            vsrc += [virtual_src]
            vdst += [virtual_dst]
            for child in topic_hier[virtual_src]:
                hsrc += [child, virtual_dst] 
                hdst += [virtual_dst, child]

        downward_adjmat = dgl.graph((torch.tensor(vsrc), torch.tensor(vdst)), num_nodes=num_topics)
        upward_adjmat = dgl.graph((torch.tensor(vdst), torch.tensor(vsrc)), num_nodes=num_topics)
        sideward_adjmat = dgl.graph((torch.tensor(hsrc), torch.tensor(hdst)), num_nodes=num_topics)

        downward_adjmat = dgl.add_self_loop(downward_adjmat)
        upward_adjmat = dgl.add_self_loop(upward_adjmat)

        return downward_adjmat, upward_adjmat, sideward_adjmat

    def to_device(self, device):
        """Move all model components to the specified device"""
        super().to(device)  # Call parent's to() to handle nn.Module parameters
        
        # Move tensors
        self.topic_node_feats = self.topic_node_feats.to(device)
        self.topic_mask_feats = self.topic_mask_feats.to(device)
        self.downward_adjmat = self.downward_adjmat.to(device)
        self.upward_adjmat = self.upward_adjmat.to(device)
        self.sideward_adjmat = self.sideward_adjmat.to(device)
        
        # Verify all components are on correct device
        print(f"TopicEncoder devices:")
        print(f"- topic_node_feats: {self.topic_node_feats.device}")
        print(f"- topic_mask_feats: {self.topic_mask_feats.device}")
        print(f"- downward_adjmat: {self.downward_adjmat.device}")
        print(f"- upward_adjmat: {self.upward_adjmat.device}")
        print(f"- sideward_adjmat: {self.sideward_adjmat.device}")
        for i, layer in enumerate(self.downward_layers):
            for name, param in layer.named_parameters():
                print(f"- downward_layers[{i}].{name}: {param.device}")
        
        return self

    def forward(self, downward_adjmat, upward_adjmat, sideward_adjmat, features):
        device = next(self.parameters()).device  # Get device from model parameters
        
        # Ensure input tensors are on the same device as model
        downward_adjmat = downward_adjmat.to(device)
        upward_adjmat = upward_adjmat.to(device)
        sideward_adjmat = sideward_adjmat.to(device)
        features = features.to(device)
        
        h = features
        
        # Apply GNN layers
        for layer_idx in range(self.num_layers):
            # Move tensors to same device as layer weights
            layer_device = next(self.downward_layers[layer_idx].parameters()).device
            downward_h = self.downward_layers[layer_idx](downward_adjmat.to(layer_device), h.to(layer_device))
            upward_h = self.upward_layers[layer_idx](upward_adjmat.to(layer_device), h.to(layer_device))
            sideward_h = self.sideward_layers[layer_idx](sideward_adjmat.to(layer_device), h.to(layer_device))
            
            # Combine the outputs
            h = downward_h + upward_h + sideward_h
            h = self.activation(h)
            
        return h

    def encode(self, use_mask=True):
        device = next(self.parameters()).device  # Get device from model parameters
        
        # Ensure input tensors are on the same device as model
        topic_node_feats = self.topic_node_feats.to(device)
        topic_mask_feats = self.topic_mask_feats.to(device)
        
        # Create mask on same device
        if use_mask:
            topic_mask = torch.rand(topic_node_feats.shape[0], 1, device=device) < 0.15
            topic_node_feats = topic_mask * topic_mask_feats + (~topic_mask) * topic_node_feats

        # Ensure adjacency matrices are on same device
        downward_adjmat = self.downward_adjmat.to(device)
        upward_adjmat = self.upward_adjmat.to(device)
        sideward_adjmat = self.sideward_adjmat.to(device)
        
        h = self.forward(downward_adjmat, upward_adjmat, sideward_adjmat, topic_node_feats)
        return h

    def inductive_encode(self):
        parent2virtualh = {}
        virtual_id = self.num_topics
        topic_node_feats = torch.cat([self.topic_node_feats, self.topic_mask_feats[None, :]], dim=0)
        for parent_id in self.topic_hier:
            downward_adjmat, upward_adjmat, sideward_adjmat = self._generate_adjmat(self.topic_hier, self.num_topics+1, parent_id, virtual_id)
            downward_adjmat = downward_adjmat.to(topic_node_feats.device)
            upward_adjmat = upward_adjmat.to(topic_node_feats.device)
            sideward_adjmat = sideward_adjmat.to(topic_node_feats.device)
            h = self.forward(downward_adjmat, upward_adjmat, sideward_adjmat, topic_node_feats)
            parent2virtualh[parent_id] = h[virtual_id, :]
        return parent2virtualh

    def inductive_target(self, vid2pid, novel_topic_hier):
        virtual2target = {}
        for virtual_id, parent_id in vid2pid.items():
            target = torch.zeros(self.num_topics)
            for novel_topic_id in novel_topic_hier[parent_id]:
                target[novel_topic_id] = 1
            virtual2target[virtual_id] = target
        return virtual2target

"""
    3. Topic-Document Similarity Predictor
"""
class BilinearInteraction(nn.Module):
    def __init__(self, doc_dim, topic_dim, num_topics=None, bias=True):
        super(BilinearInteraction, self).__init__()
        self.weight = Parameter(torch.Tensor(doc_dim, topic_dim))
        self.use_bias = bias
        if self.use_bias:
            self.bias = Parameter(torch.Tensor(num_topics))

        bound = 1.0 / math.sqrt(doc_dim)
        init.uniform_(self.weight, -bound, bound)
        if self.use_bias:
            init.uniform_(self.bias, -bound, bound)

    def forward(self, e1, e2):
        """
        e1: tensor of size (batch_size, doc_dim)
        e2: tensor of size (num_topics, topic_dim)
        return: tensor of size (batch_size, num_topics)
        """
        scores = torch.matmul(torch.matmul(e1, self.weight), e2.T)
        if self.use_bias:
            scores = scores + self.bias
        return scores

    def compute_attn_scores(self, e1, e2):
        """
        e1: tensor of size (batch_size, num_tokens, doc_dim)
        e2: tensor of size (batch_size, topic_dim)
        return: tensor of size (batch_size, num_toknes)
        """
        scores = torch.bmm(torch.matmul(e1, self.weight), e2.unsqueeze(dim=2))
        scores = scores.squeeze()
        return scores

"""
    4. Topic-conditional Phrase Generator
"""
class TransformerPhraseDecoder(BaseModel):
    def __init__(self, input_embeddings, pad_token_id, bos_token_id, eos_token_id, num_layers, num_heads, max_length, use_flash_attention=False):
        super().__init__()
        self.vocab_size, self.hidden_size = input_embeddings.word_embeddings.weight.shape
        self.input_embeddings = input_embeddings
        self.output_embeddings = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        model_layer = TransformerDecoderLayer(
            d_model=self.hidden_size, 
            nhead=num_heads, 
            batch_first=True
        )
        self.model = TransformerDecoder(model_layer, num_layers=num_layers)

        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def _make_causal_mask(self, x):
        # Create causal mask for decoder
        sz = x.size(1)
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(x.device)

    def forward(self, x, decoder_context=None):
        # Handle BERT-style dictionary input
        if isinstance(x, dict):
            input_ids = x['input_ids']
            token_type_ids = x.get('token_type_ids', None)
            # Only pass input_ids and token_type_ids to embeddings
            x = self.input_embeddings(
                input_ids=input_ids,
                token_type_ids=token_type_ids
            )
        else:
            x = self.input_embeddings(x)
        
        # Create attention masks
        attn_mask = self._make_causal_mask(x)
        # Use attention_mask from input if provided
        if isinstance(x, dict) and 'attention_mask' in x:
            padding_mask = ~x['attention_mask'].bool()
        else:
            padding_mask = None
        
        # Run through transformer decoder
        output = self.model(
            x,
            decoder_context,
            tgt_mask=attn_mask,
            memory_mask=None,
            tgt_key_padding_mask=padding_mask,
            memory_key_padding_mask=None
        )
        
        # Project to vocabulary
        output = self.output_embeddings(output)
        
        return output

    def generate(self, context, beam_size=5, length_penalty=1.0):
        batch_size = context.size(0)
        device = context.device
        
        # Initialize beam state
        sequences = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=device)
        sequence_scores = torch.zeros(batch_size, 1, device=device)
        
        # Store finished sequences and their scores
        finished_sequences = [[] for _ in range(batch_size)]
        finished_scores = [[] for _ in range(batch_size)]
        
        for step in range(self.max_length - 1):
            # Get logits for next token
            logits = self.forward(sequences, context)  # [batch_size * beam_size, seq_len, vocab_size]
            curr_logits = logits[:, -1, :]  # [batch_size * beam_size, vocab_size]
            
            # Get log probabilities
            log_probs = F.log_softmax(curr_logits, dim=-1)  # [batch_size * beam_size, vocab_size]
            vocab_size = log_probs.size(-1)
            
            if step == 0:
                # For first step, only use first beam
                sequence_scores = sequence_scores + log_probs  # [batch_size, vocab_size]
                scores, indices = sequence_scores.topk(beam_size, dim=-1)  # [batch_size, beam_size]
                
                # Convert indices to next tokens
                next_tokens = indices
                
                # Update sequences
                sequences = sequences.repeat(1, beam_size).view(-1, sequences.size(-1))  # [batch_size * beam_size, seq_len]
                sequences = torch.cat([sequences, next_tokens.view(-1, 1)], dim=-1)
                sequence_scores = scores.view(-1, 1)
                
                # Update context for beam size
                context = context.repeat_interleave(beam_size, dim=0)
            else:
                num_beams = sequences.size(0) // batch_size
                
                # Calculate scores for each beam and vocab item
                sequence_scores = sequence_scores.unsqueeze(-1)  # [batch_size * beam_size, 1]
                vocab_scores = log_probs  # [batch_size * beam_size, vocab_size]
                
                # Compute length penalty
                length_penalty_score = ((5 + step + 1) / 6) ** length_penalty
                
                # Reshape scores for topk
                curr_scores = (sequence_scores + vocab_scores) / length_penalty_score  # [batch_size * beam_size, vocab_size]
                curr_scores = curr_scores.view(batch_size, num_beams * vocab_size)
                
                # Select top beams and their tokens
                scores, indices = curr_scores.topk(beam_size, dim=-1)  # [batch_size, beam_size]
                beam_indices = indices // vocab_size  # [batch_size, beam_size]
                token_indices = indices % vocab_size  # [batch_size, beam_size]
                
                # Compute offsets for gather operation
                beam_indices = beam_indices + (torch.arange(batch_size, device=device) * num_beams).unsqueeze(-1)
                
                # Gather sequences
                sequences = sequences.view(-1, sequences.size(-1))  # [batch_size * beam_size, seq_len]
                sequences = sequences[beam_indices.view(-1)]  # [batch_size * beam_size, seq_len]
                sequences = torch.cat([sequences, token_indices.view(-1, 1)], dim=-1)
                
                # Update scores
                sequence_scores = scores.view(-1, 1)
            
            # Check for completed sequences
            eos_mask = sequences[:, -1] == self.eos_token_id
            
            if eos_mask.any():
                # Add finished sequences to their respective batch lists
                for idx in range(sequences.size(0)):
                    if eos_mask[idx]:
                        batch_idx = idx // beam_size
                        finished_sequences[batch_idx].append(sequences[idx].tolist())
                        finished_scores[batch_idx].append(sequence_scores[idx].item())
                
                # Remove finished sequences
                if eos_mask.all():
                    break
                    
                non_finished_mask = ~eos_mask
                sequences = sequences[non_finished_mask]
                sequence_scores = sequence_scores[non_finished_mask]
                context = context[non_finished_mask]
                
                if sequences.size(0) == 0:  # All sequences finished
                    break
        
        # Handle any unfinished sequences
        if sequences.size(0) > 0:
            for idx in range(sequences.size(0)):
                batch_idx = idx // beam_size
                finished_sequences[batch_idx].append(sequences[idx].tolist())
                finished_scores[batch_idx].append(sequence_scores[idx].item())
        
        # Select the best sequence for each batch
        results = []
        for batch_seqs, batch_scores in zip(finished_sequences, finished_scores):
            if not batch_seqs:  # If no sequences finished, use a default sequence
                results.append(torch.tensor([self.bos_token_id, self.eos_token_id], device=device))
            else:
                # Sort by score and take the best one
                best_idx = np.argmax(batch_scores)
                results.append(torch.tensor(batch_seqs[best_idx], device=device))
        
        # Pad sequences to same length
        max_len = max(len(seq) for seq in results)
        padded_results = []
        for seq in results:
            padding = torch.full((max_len - len(seq),), self.pad_token_id, device=device)
            padded_results.append(torch.cat([seq, padding]))
        
        return torch.stack(padded_results)