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
class Interaction(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.temperature = nn.Parameter(torch.ones(1))
        self.interaction = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def forward(self, doc_embed, topic_embed):
        # Transform document embeddings
        doc_embed = self.interaction(doc_embed)
        
        # Scale dot product by sqrt(dim) and learnable temperature
        sim = torch.matmul(doc_embed, topic_embed.t()) 
        sim = sim * F.softplus(self.temperature)
        
        return sim

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
        # Tie output projection with input embeddings
        self.output_layer = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.output_layer.weight = input_embeddings.word_embeddings.weight
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
            # Only allow word pieces that start with ## or are full words
            is_valid_wordpiece = not token.startswith('##') or token == '[UNK]'
            # Check if token is ASCII and not punctuation/special chars
            is_ascii = all(ord(c) < 128 and c.isalnum() for c in token.replace('##', ''))
            self.valid_tokens[i] = (is_special or (is_ascii and is_valid_wordpiece))

        # Always allow special tokens
        self.valid_tokens[self.pad_token_id] = True
        self.valid_tokens[self.bos_token_id] = True
        self.valid_tokens[self.eos_token_id] = True

    def _make_causal_mask(self, x):
        sz = x.size(1)
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, input_ids, context):
        if isinstance(input_ids, dict):
            x = self.input_embeddings(input_ids['input_ids'])  # [batch, seq_len, embed_dim]
        else:
            x = self.input_embeddings(input_ids)  # [batch, seq_len, embed_dim]

        # Create attention mask
        tgt_mask = self._make_causal_mask(x).to(x.device)
        
        # Ensure context has sequence dimension
        if context.dim() == 2:
            context = context.unsqueeze(1)  # [batch, 1, hidden_size]
        
        # Forward through transformer
        output = self.model(
            x,
            context,
            tgt_mask=tgt_mask,
            tgt_is_causal=True
        )
        
        # Project to vocabulary
        output = self.output_layer(output)
        
        return output

    def generate(self, context, attention_mask):
        """
        Generate phrases using document attention
        context: document embeddings [batch, seq, hidden]
        attention_mask: document attention mask [batch, seq]
        """
        batch_size = context.size(0)
        
        # Start with CLS token
        current_token = torch.full((batch_size, 1), self.tokenizer.cls_token_id, 
                                 dtype=torch.long, device=context.device)
        
        # Project and normalize context
        context_proj = self.context_proj(context)  # [batch, seq, hidden]
        context_norm = self.context_norm(context_proj)
        
        # Move valid_tokens to correct device
        valid_tokens = self.valid_tokens.to(context.device)
        
        # Generation parameters
        min_tokens = 3  # Minimum tokens after CLS
        max_tokens = 10  # Maximum total tokens
        repetition_penalty = 2.0
        temperature = 0.7  # Lower temperature for more focused sampling
        
        # Track generated n-grams for repetition penalty
        generated_ngrams = [{} for _ in range(batch_size)]
        
        for step in range(max_tokens - 2):  # -2 for CLS and SEP
            # Get decoder output
            decoder_output = self.forward(current_token, context_norm)  # [batch, seq, vocab]
            next_token_logits = decoder_output[:, -1, :] / temperature  # [batch, vocab]
            
            # Calculate attention scores with document tokens
            attn_scores = torch.matmul(next_token_logits, context_norm.transpose(-1, -2))  # [batch, seq]
            attn_scores = attn_scores.masked_fill(~attention_mask.bool(), -float('inf'))
            attn_probs = F.softmax(attn_scores, dim=-1)
            
            # Get document token probabilities
            doc_token_ids = context['input_ids']  # [batch, seq]
            next_token_probs = torch.zeros_like(next_token_logits)  # [batch, vocab]
            next_token_probs.scatter_add_(-1, doc_token_ids, attn_probs)
            
            # Apply repetition penalty
            for i in range(batch_size):
                # Penalize seen tokens
                for token_id in set(current_token[i].tolist()):
                    next_token_probs[i, token_id] /= repetition_penalty
                
                # Penalize completing seen trigrams
                if current_token.size(1) >= 2:
                    prev_tokens = current_token[i, -2:].tolist()
                    for token_id in range(next_token_probs.size(-1)):
                        if tuple(prev_tokens + [token_id]) in generated_ngrams[i]:
                            next_token_probs[i, token_id] /= (repetition_penalty * 2)
            
            # Mask invalid tokens
            next_token_probs = next_token_probs.masked_fill(~valid_tokens, 0)
            next_token_probs = next_token_probs.masked_fill(torch.isnan(next_token_probs), 0)
            
            # Mask special tokens except SEP when appropriate
            next_token_probs[:, self.pad_token_id] = 0
            next_token_probs[:, self.bos_token_id] = 0
            if step < min_tokens:
                next_token_probs[:, self.eos_token_id] = 0
            
            # Normalize probabilities
            next_token_probs = F.softmax(next_token_probs, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(next_token_probs, num_samples=1)
            
            # Update n-gram tracking
            for i in range(batch_size):
                if current_token.size(1) >= 2:
                    prev_tokens = current_token[i, -2:].tolist()
                    new_token = next_token[i].item()
                    trigram = tuple(prev_tokens + [new_token])
                    generated_ngrams[i][trigram] = generated_ngrams[i].get(trigram, 0) + 1
            
            # Append next token
            current_token = torch.cat([current_token, next_token], dim=1)
            
            # Stop if all sequences have generated SEP token
            if (next_token == self.eos_token_id).all():
                break
        
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
        """
        Forward pass for training
        encoder_input: dict of BERT inputs for document
        decoder_input: dict of BERT inputs for target phrase
        topic_ids: tensor of topic IDs
        """
        # Get document embeddings
        doc_outputs = self.bert(**encoder_input)
        doc_embeds = doc_outputs[0]  # [batch, seq, hidden]
        doc_mask = encoder_input['attention_mask']
        
        # Get normalized topic embeddings
        topic_embeds = self.topic_norm(self.topic_embeddings)  # [num_topics, hidden]
        
        # Calculate similarity scores
        if topic_ids is not None:
            # For training, only compute similarity with target topics
            batch_size = doc_embeds.size(0)
            doc_mean = doc_embeds.mean(dim=1)  # [batch, hidden]
            topic_embeds_selected = topic_embeds[topic_ids]  # [batch, hidden]
            
            # Calculate similarity scores with temperature
            sim_scores = torch.matmul(doc_mean, topic_embeds.t()) / self.sim_temperature
        else:
            # For inference, compute similarity with all topics
            doc_mean = doc_embeds.mean(dim=1)  # [batch, hidden]
            sim_scores = torch.matmul(doc_mean, topic_embeds.t()) / self.sim_temperature
        
        # Generate or decode phrases
        if decoder_input is not None:
            # Training mode - use teacher forcing
            gen_scores = self.decoder(decoder_input['input_ids'], doc_embeds)
        else:
            # Inference mode - generate new phrases
            gen_scores = self.decoder.generate(doc_embeds, doc_mask)
            
        return sim_scores, gen_scores

    def gen(self, encoder_input, topic_ids):
        # Get normalized topic embeddings
        normalized_topics = self.topic_norm(self.topic_embeddings)
        topic_context = normalized_topics[topic_ids]
        
        # Generate phrases
        return self.decoder.generate(topic_context, encoder_input['attention_mask'])