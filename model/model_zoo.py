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
class GCNLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, norm='right'):
        super().__init__()
        self.conv = GraphConv(in_dim, hidden_dim, norm=norm, allow_zero_in_degree=True)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, g, x):
        identity = x
        out = self.conv(g, x)
        out = out + identity  # residual connection
        out = self.norm(out)  # layer normalization
        out = F.gelu(out)     # non-linearity
        return out

class GCNTopicEncoder(BaseModel):
    def __init__(self, topic_hier, topic_node_feats, topic_mask_feats, topic_num_layers): 
        super(GCNTopicEncoder, self).__init__()
        num_topics, topic_embed_dim = topic_node_feats.shape
        in_dim, hidden_dim, out_dim = topic_embed_dim, topic_embed_dim, topic_embed_dim

        self.num_layers = topic_num_layers
        
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

        # Create GCN layers with residual connections
        self.downward_layers = nn.ModuleList([GCNLayer(in_dim if i == 0 else hidden_dim, 
                                                      out_dim if i == topic_num_layers-1 else hidden_dim) 
                                            for i in range(topic_num_layers)])
        self.upward_layers = nn.ModuleList([GCNLayer(in_dim if i == 0 else hidden_dim,
                                                    out_dim if i == topic_num_layers-1 else hidden_dim)
                                          for i in range(topic_num_layers)])
        self.sideward_layers = nn.ModuleList([GCNLayer(in_dim if i == 0 else hidden_dim,
                                                      out_dim if i == topic_num_layers-1 else hidden_dim)
                                            for i in range(topic_num_layers)])
        
        # Output layer norm
        self.output_norm = nn.LayerNorm(out_dim)
        
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
            h = F.gelu(h)
            
        h = self.output_norm(h)
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
class CrossAttentionInteraction(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Initialize with smaller values to prevent saturation
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        # Initialize weights with smaller values
        nn.init.xavier_uniform_(self.query.weight, gain=0.1)
        nn.init.xavier_uniform_(self.key.weight, gain=0.1)
        nn.init.xavier_uniform_(self.value.weight, gain=0.1)
        
        # Add layer norm for better training dynamics
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, doc_tensor, topic_encoder_output):
        """
        doc_tensor: [batch_size, hidden_size]
        topic_encoder_output: [num_topics, hidden_size]
        """
        batch_size = doc_tensor.size(0)
        num_topics = topic_encoder_output.size(0)
        
        # Apply layer norm
        doc_tensor = self.layer_norm(doc_tensor)
        topic_encoder_output = self.layer_norm(topic_encoder_output)
        
        # Project inputs
        q = self.query(doc_tensor)  # [batch_size, hidden_size]
        k = self.key(topic_encoder_output)  # [num_topics, hidden_size]
        v = self.value(topic_encoder_output)  # [num_topics, hidden_size]
        
        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(0, 1))  # [batch_size, num_topics]
        attention_scores = attention_scores / math.sqrt(self.hidden_size)  # Scale scores
        
        # Apply softmax row-wise (over topics)
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, num_topics]
        
        print(f"[DEBUG] Raw attention scores min/max/mean: {attention_scores.min().item():.3f}/{attention_scores.max().item():.3f}/{attention_scores.mean().item():.3f}")
        print(f"[DEBUG] Attention weights min/max/mean: {attention_weights.min().item():.3f}/{attention_weights.max().item():.3f}/{attention_weights.mean().item():.3f}")
        
        # Compute similarity scores using cosine similarity
        doc_norm = F.normalize(doc_tensor, p=2, dim=1)  # [batch_size, hidden_size]
        topic_norm = F.normalize(topic_encoder_output, p=2, dim=1)  # [num_topics, hidden_size]
        sim_score = torch.matmul(doc_norm, topic_norm.transpose(0, 1))  # [batch_size, num_topics]
        
        print(f"[DEBUG] Similarity scores min/max/mean: {sim_score.min().item():.3f}/{sim_score.max().item():.3f}/{sim_score.mean().item():.3f}")
        print(f"[DEBUG] sim_score shape: {sim_score.shape}")
        
        return sim_score

class ContextCombiner(nn.Module):
    def __init__(self, doc_dim, topic_dim):
        super().__init__()
        self.doc_dim = doc_dim
        self.topic_dim = topic_dim
        
        # Cross attention for doc-topic interaction
        self.cross_attn = nn.MultiheadAttention(doc_dim, num_heads=8, batch_first=True)
        
        # FFN for combining contexts
        self.linear1 = nn.Linear(doc_dim + topic_dim, doc_dim)
        self.linear2 = nn.Linear(doc_dim, doc_dim)
        self.norm1 = nn.LayerNorm(doc_dim)
        self.norm2 = nn.LayerNorm(doc_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, topic_ctx, doc_ctx):
        """
        topic_ctx: (batch_size, topic_dim)
        doc_ctx: (batch_size, seq_len, doc_dim)
        """
        batch_size = doc_ctx.shape[0]
        
        # Expand topic context to match doc context sequence length
        topic_ctx = topic_ctx.unsqueeze(1)  # (batch_size, 1, topic_dim)
        
        # Cross attention between doc and topic
        attn_out, _ = self.cross_attn(
            query=doc_ctx,
            key=topic_ctx,
            value=topic_ctx
        )
        
        # Combine with original doc context
        combined = torch.cat([attn_out, doc_ctx], dim=-1)
        
        # Two-layer feed-forward with residual
        out = self.linear1(combined)
        out = F.gelu(out)
        out = self.dropout(out)
        out = self.norm1(out)
        
        residual = out
        out = self.linear2(out)
        out = F.gelu(out)
        out = self.dropout(out)
        out = residual + out
        out = self.norm2(out)
        
        return out

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

    def generate(self, context):
        batch_size = context.size(0)
        current_token = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=context.device)
        
        for _ in range(self.max_length - 1):
            logits = self.forward(current_token, context)
            next_token = logits[:, -1:].argmax(dim=-1)
            current_token = torch.cat([current_token, next_token], dim=1)
            
            if (next_token == self.eos_token_id).all():
                break
                
        return current_token