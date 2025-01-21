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
from transformers import AutoModel, AutoConfig, AutoTokenizer
from base import BaseModel

"""
    1. Document Encoder
"""
class BertDocEncoder(BaseModel):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
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
    def __init__(self, input_embeddings, tokenizer, pad_token_id, bos_token_id, eos_token_id, num_layers, num_heads, max_length, use_flash_attention=False):
        super().__init__()
        self.vocab_size, self.hidden_size = input_embeddings.word_embeddings.weight.shape
        self.input_embeddings = input_embeddings
        self.output_embeddings = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.tokenizer = tokenizer
        
        # Add context projection and layer norm
        self.context_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.context_norm = nn.LayerNorm(self.hidden_size)
        
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
        
        # Create valid token mask (only allow ASCII characters and special tokens)
        self.valid_tokens = torch.zeros(self.vocab_size, dtype=torch.bool)
        for i in range(self.vocab_size):
            token = self.tokenizer.convert_ids_to_tokens(i)
            # Allow token if it's ASCII or a special token
            is_special = token.startswith('[') and token.endswith(']')
            is_ascii = all(ord(c) < 128 for c in token)
            self.valid_tokens[i] = is_special or is_ascii

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
            x = self.input_embeddings(
                input_ids=input_ids,
                token_type_ids=token_type_ids
            )
        else:
            x = self.input_embeddings(x)
        
        # Project and normalize context
        if decoder_context is not None:
            decoder_context = self.context_norm(self.context_proj(decoder_context))
        
        # Create attention masks
        attn_mask = self._make_causal_mask(x)
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
        
        # Debug print
        print(f"\n[DEBUG] Generation start:")
        print(f"Batch size: {batch_size}")
        print(f"Context shape: {context.shape}")
        print(f"Initial token shape: {current_token.shape}")
        
        # Project and normalize context
        context = self.context_norm(self.context_proj(context))
        
        # Initialize sampling parameters
        temperature = 0.8
        top_p = 0.9
        min_tokens = 3  # Minimum number of tokens to generate
        
        # Move valid_tokens to correct device and expand for batch
        valid_tokens = self.valid_tokens.to(context.device)
        valid_tokens = valid_tokens.unsqueeze(0).expand(batch_size, -1)
        
        # Debug print
        print(f"Valid tokens shape: {valid_tokens.shape}")
        print(f"Number of valid tokens: {valid_tokens.sum().item()}")
        
        generated_tokens = []
        
        for step in range(self.max_length - 1):
            # Get logits from decoder
            logits = self.forward(current_token, context)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Debug shapes
            if step == 0:
                print(f"\nStep {step} shapes:")
                print(f"Logits shape: {logits.shape}")
                print(f"Next token logits shape: {next_token_logits.shape}")
            
            # Mask out PAD token (prefer SEP for ending)
            next_token_logits[:, self.pad_token_id] = float('-inf')
            
            # Apply vocabulary filtering
            next_token_logits = next_token_logits.masked_fill(~valid_tokens, float('-inf'))
            
            # Apply nucleus sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Keep at least min_tokens before allowing EOS
            if step < min_tokens:
                next_token_logits[:, self.eos_token_id] = float('-inf')
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from filtered distribution
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(next_token_probs, num_samples=1)
            
            # Debug first token
            if step == 0:
                print(f"\nFirst token generation:")
                print(f"Selected token IDs: {next_token.tolist()}")
                print(f"Selected tokens: {[self.tokenizer.convert_ids_to_tokens(t.item()) for t in next_token]}")
            
            current_token = torch.cat([current_token, next_token], dim=1)
            generated_tokens.append(next_token)
            
            # Stop if SEP token is generated
            if (next_token == self.eos_token_id).any():
                # Replace everything after SEP with SEP
                for b in range(batch_size):
                    if next_token[b] == self.eos_token_id:
                        current_token[b, -1] = self.eos_token_id
                break
    
        # Post-process: ensure sequences start with CLS and end with SEP
        for b in range(batch_size):
            # Find first SEP token
            sep_pos = (current_token[b] == self.eos_token_id).nonzero()
            if len(sep_pos) > 0:
                # Keep only up to first SEP
                current_token[b, sep_pos[0]+1:] = self.eos_token_id
            else:
                # If no SEP, append it
                current_token[b, -1] = self.eos_token_id
        
            # Ensure starts with CLS
            current_token[b, 0] = self.bos_token_id
    
        # Debug final output
        print(f"\nFinal generation:")
        print(f"Output shape: {current_token.shape}")
        for b in range(min(2, batch_size)):  # Show first 2 examples
            tokens = [self.tokenizer.convert_ids_to_tokens(t.item()) for t in current_token[b]]
            print(f"Sample {b}: {' '.join(tokens)}")
    
        return current_token

"""
    5. Topic Expansion Model
"""
class TopicExpansionModel(BaseModel):
    def __init__(self, bert_model_name, num_topics, max_length, use_flash_attention=False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.num_topics = num_topics
        
        # Add topic embeddings with normalization
        self.topic_embeddings = nn.Parameter(torch.randn(num_topics, self.bert.config.hidden_size))
        self.topic_norm = nn.LayerNorm(self.bert.config.hidden_size)
        
        # Add similarity temperature parameter (learnable)
        self.sim_temperature = nn.Parameter(torch.ones(1))
        
        # Initialize decoder
        self.decoder = TransformerPhraseDecoder(
            input_embeddings=self.bert.embeddings,
            tokenizer=self.tokenizer,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.cls_token_id,
            eos_token_id=self.tokenizer.sep_token_id,
            num_layers=6,
            num_heads=12,
            max_length=max_length,
            use_flash_attention=use_flash_attention
        )

    def forward(self, encoder_input, decoder_input=None, topic_ids=None):
        # Get document embeddings
        encoder_output = self.bert(**encoder_input).last_hidden_state[:, 0]  # Use CLS token
        
        # Normalize topic embeddings
        normalized_topics = self.topic_norm(self.topic_embeddings)
        normalized_docs = F.normalize(encoder_output, dim=-1)
        
        # Calculate similarity scores with temperature
        sim_scores = torch.matmul(normalized_docs, normalized_topics.t())
        sim_scores = sim_scores / self.sim_temperature
        
        if decoder_input is not None and topic_ids is not None:
            # Get topic embeddings for the target topics
            topic_context = normalized_topics[topic_ids]
            
            # Generate phrases
            gen_scores = self.decoder(decoder_input, topic_context)
            return sim_scores, gen_scores
        
        return sim_scores

    def gen(self, encoder_input, topic_ids):
        # Get normalized topic embeddings
        normalized_topics = self.topic_norm(self.topic_embeddings)
        topic_context = normalized_topics[topic_ids]
        
        # Generate phrases
        return self.decoder.generate(topic_context)