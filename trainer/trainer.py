import numpy as np
import torch
import torch.nn.functional as F
from base import BaseTrainer
from utils import MetricTracker
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import euclidean_distances
from gensim.models import KeyedVectors
import os, pickle
import time
from sentence_transformers import SentenceTransformer
import json

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterions, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None):
        super().__init__(model, criterions, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.dataset = data_loader.dataset

        self.len_epoch = len(self.data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.log_step = int(len(data_loader.dataset) / data_loader.batch_size * 0.2)

        self.train_metrics = MetricTracker('loss', 'sim_loss', 'gen_loss', writer=self.writer)
        self.valid_metrics = MetricTracker('loss', 'sim_loss', 'gen_loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        
        # Setup logging
        os.makedirs('logs', exist_ok=True)
        current_time = time.strftime("%Y%m%d-%H%M%S")
        self.log_file = os.path.join('logs', f'training_log_{current_time}.txt')
        self.log_info(f"Starting training at {current_time}")

    def log_info(self, message):
        """Log message to both console and file"""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        
        # Loss weights - give more weight to generation loss
        sim_weight = 0.3
        gen_weight = 0.7
        
        # Gradient clipping norm
        max_grad_norm = 1.0

        for batch_idx, batch_data in enumerate(self.data_loader):
            doc_ids, doc_infos, topic_ids, phrase_infos = batch_data
            
            # Debug: Check input shapes and values
            if batch_idx == 0:  # Only print for first batch
                debug_info = "\n[DEBUG] Batch Information:"
                debug_info += f"\nDocument IDs shape: {doc_ids.shape}"
                debug_info += f"\nTopic IDs shape: {topic_ids.shape}"
                debug_info += f"\nUnique topics in batch: {topic_ids.unique().tolist()}"
                debug_info += f"\nInput sequence lengths: {doc_infos['attention_mask'].sum(1).tolist()}"
                debug_info += f"\nTarget phrase lengths: {phrase_infos['attention_mask'].sum(1).tolist()}"
                self.log_info(debug_info)
            
            encoder_input = {k: v.to(self.device) for k, v in doc_infos.items()}
            # Keep CLS token in both input and target
            decoder_input = {k: v[:, :-1].to(self.device) for k, v in phrase_infos.items()}
            gen_target = phrase_infos['input_ids'][:, :-1].to(self.device)  # Keep CLS, remove last token for target
            
            sim_target = topic_ids.to(self.device)

            self.optimizer.zero_grad()
            
            sim_score, gen_score = self.model(encoder_input, decoder_input, topic_ids)
            
            # Debug: Check model outputs
            if batch_idx == 0:  # Only print for first batch
                debug_info = "\n[DEBUG] Model Outputs:"
                debug_info += f"\nSimilarity scores shape: {sim_score.shape}"
                debug_info += f"\nScore range: [{sim_score.min():.3f}, {sim_score.max():.3f}]"
                debug_info += f"\nGeneration logits shape: {gen_score.shape}"
                
                # Print predicted vs actual topics
                pred_topics = sim_score.argmax(dim=1)
                debug_info += "\n\nTopic Prediction Sample:"
                for i in range(min(3, len(pred_topics))):
                    debug_info += f"\nPred: {pred_topics[i].item()} | True: {sim_target[i].item()}"
                
                # Print sample phrase
                if hasattr(self.dataset, 'bert_tokenizer'):
                    gen_tokens = gen_score[0].argmax(dim=1)
                    generated = self.dataset.bert_tokenizer.decode(gen_tokens)
                    target = self.dataset.bert_tokenizer.decode(gen_target[0])
                    debug_info += "\n\nPhrase Generation Sample:"
                    debug_info += f"\nGenerated: {generated}"
                    debug_info += f"\nTarget: {target}"
                self.log_info(debug_info)
            
            sim_loss = self.criterions['sim'](sim_score, sim_target)
            gen_loss = self.criterions['gen'](gen_score.view(-1, gen_score.size(-1)), gen_target.view(-1))
            
            # Apply loss weights and ensure they're positive
            loss = sim_weight * sim_loss + gen_weight * gen_loss
            
            # Debug: Check loss values and predictions
            if batch_idx % 100 == 0:
                debug_info = "\n[DEBUG] Loss Values:"
                debug_info += f"\nSimilarity Loss (raw): {sim_loss.item():.4f}"
                debug_info += f"\nGeneration Loss (raw): {gen_loss.item():.4f}"
                debug_info += f"\nWeighted Sim Loss: {(sim_weight * sim_loss).item():.4f}"
                debug_info += f"\nWeighted Gen Loss: {(gen_weight * gen_loss).item():.4f}"
                debug_info += f"\nTotal Loss: {loss.item():.4f}"
                
                # Print sample predictions
                with torch.no_grad():
                    # Topic prediction accuracy
                    pred_topics = sim_score.argmax(dim=1)
                    topic_acc = (pred_topics == sim_target).float().mean()
                    debug_info += f"\nTopic Prediction Accuracy: {topic_acc.item():.4f}"
                    
                    # Generation perplexity
                    gen_perplexity = torch.exp(gen_loss)
                    debug_info += f"\nGeneration Perplexity: {gen_perplexity.item():.4f}"
                    
                    # Print sample generations
                    debug_info += "\n\nSample Generations:"
                    for i in range(min(3, len(gen_score))):
                        gen_tokens = gen_score[i].argmax(dim=1)
                        generated = self.dataset.bert_tokenizer.decode(gen_tokens)
                        target = self.dataset.bert_tokenizer.decode(gen_target[i])
                        debug_info += f"\nGenerated {i}: {generated}"
                        debug_info += f"\nTarget {i}:   {target}\n"
                
                self.log_info(debug_info)

            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('sim_loss', sim_loss.item())
            self.train_metrics.update('gen_loss', gen_loss.item())

            if batch_idx % self.log_step == 0:
                self.log_info('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        if self.do_validation:
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            self.log_info(f'[{current_time}] Starting validation for epoch: {epoch}')
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.valid_data_loader):
                doc_ids, doc_infos, topic_ids, phrase_infos = batch_data
        
                encoder_input = {k: v.to(self.device) for k, v in doc_infos.items()}
                decoder_input = {k: v[:, :-1].to(self.device) for k, v in phrase_infos.items()}

                sim_target = topic_ids.to(self.device)
                gen_target = phrase_infos['input_ids'][:, :-1].to(self.device)  # Keep CLS, remove last token for target

                # Get model outputs
                sim_score, gen_score = self.model(encoder_input, decoder_input, topic_ids)
                
                # Calculate losses
                sim_loss = self.criterions['sim'](sim_score, sim_target)
                gen_loss = self.criterions['gen'](gen_score, gen_target)
                loss = sim_loss + gen_loss

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('sim_loss', sim_loss.item()) 
                self.valid_metrics.update('gen_loss', gen_loss.item())
                self.valid_metrics.update('loss', loss.item())      
                
                # Evaluate generation quality
                if batch_idx == 0:  # Only for first batch
                    # Generate phrases
                    gen_output = self.model.gen(encoder_input, topic_ids)
                    
                    # Get attention masks
                    output_mask = (gen_output != self.dataset.bert_tokenizer.pad_token_id).float()
                    target_mask = (gen_target != self.dataset.bert_tokenizer.pad_token_id).float()
                    
                    # Calculate metrics
                    for met in self.metric_ftns:
                        if met.__name__ == 'embedding_sim':
                            met_val = met(gen_output, gen_target, output_mask, target_mask)
                        else:
                            met_val = met(gen_score, gen_target)
                        self.valid_metrics.update(met.__name__, met_val)
                    
                    # Log sample generations
                    if hasattr(self.dataset, 'bert_tokenizer'):
                        gen_info = "\nGeneration Samples:"
                        for i in range(min(3, len(gen_output))):
                            generated = self.dataset.bert_tokenizer.decode(gen_output[i])
                            target = self.dataset.bert_tokenizer.decode(gen_target[i])
                            gen_info += f"\nTopic {topic_ids[i].item()}:"
                            gen_info += f"\nGenerated: {generated}"
                            gen_info += f"\nTarget: {target}\n"
                        self.log_info(gen_info)

                if batch_idx % self.log_step == 0:
                    self.log_info('Validation Epoch: {} {} Loss: {:.6f}'.format(
                        epoch,
                        self._progress_validation(batch_idx),
                        loss.item()))

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _progress_validation(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.valid_data_loader, 'n_samples'):
            current = batch_idx * self.valid_data_loader.batch_size
            total = self.valid_data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


    def infer(self, config, parent_vid=None, depth=0, max_depth=3):
        """Infer and expand topics recursively with improved hierarchy handling.
        
        Args:
            config: Configuration dictionary
            parent_vid: Parent vertex ID if doing recursive expansion
            depth: Current depth in hierarchy
            max_depth: Maximum depth to expand to
        """
        self.model.eval()
        dataset = self.data_loader.dataset
        
        with torch.no_grad():
            # Get topic embeddings from GCN
            parent2virtualh = self.model.topic_encoder.inductive_encode()
            
            # Initialize hierarchy with root node if starting fresh
            if not os.path.exists('discovered_topics.json'):
                output = {}
                root_name = dataset.topics["0"]  # "politics" is the root
                output[root_name] = []
            else:
                with open('discovered_topics.json', 'r') as f:
                    output = json.load(f)
            
            # Track seen topics to avoid duplicates
            seen_topics = set()
            for existing_topics in output.values():
                for topic in existing_topics:
                    seen_topics.add(topic['name'])
            
            # Process each parent topic
            for parent_id in dataset.topic_fullhier:
                # Get parent name and children
                parent_name = dataset.topics[parent_id]
                children = dataset.topic_fullhier[parent_id]
                
                # Initialize parent in output if not exists
                if parent_name not in output:
                    output[parent_name] = []
                
                # Add each child as a discovered topic and recursively expand
                for child_id in children:
                    child_name = dataset.topics[child_id]
                    if child_name not in seen_topics:
                        seen_topics.add(child_name)
                        
                        # Get parent's embedding for context
                        parent_rank = dataset.topicID2topicRank[parent_id]
                        if parent_rank in parent2virtualh:
                            parent_embed = parent2virtualh[parent_rank]
                            
                            # Generate phrases for potential subtopics
                            generated_phrases = []
                            print(f"Generating phrases for {child_name}...")
                            for i in range(20):  # Generate multiple phrases to cluster
                                # Add sequence length dimension: [batch_size, seq_len, hidden_size]
                                phrase_embed = parent_embed.unsqueeze(0).unsqueeze(1)  
                                phrase_output = self.model.phrase_decoder.generate(phrase_embed)
                                phrase = dataset.bert_tokenizer.decode(phrase_output[0], skip_special_tokens=True)
                                print(f"Generated phrase {i+1}: {phrase}")
                                if len(phrase.strip()) > 0:
                                    generated_phrases.append(phrase.strip())
                            
                            print(f"Found {len(generated_phrases)} valid phrases")
                            # Cluster phrases into subtopics
                            if len(generated_phrases) >= 5:
                                subtopics = self._cluster_phrases(generated_phrases, parent_embed)
                                print(f"Generated {len(subtopics)} subtopics")
                                
                                # Add child with its subtopics
                                child_entry = {
                                    "name": child_name,
                                    "quality_score": 1.0,  # Prewritten topics get max quality score
                                    "subtopics": []
                                }
                                
                                # Add high quality subtopics
                                for subtopic in subtopics:
                                    if subtopic["quality_score"] > 0.5:  # Only keep good subtopics
                                        child_entry["subtopics"].append(subtopic)
                                
                                output[parent_name].append(child_entry)
                            else:
                                # Add child without subtopics if we couldn't generate enough phrases
                                output[parent_name].append({
                                    "name": child_name,
                                    "quality_score": 1.0
                                })
                        else:
                            # Add child without subtopics if no parent embedding
                            output[parent_name].append({
                                "name": child_name,
                                "quality_score": 1.0
                            })
            
            # Write output
            with open('discovered_topics.json', 'w') as f:
                json.dump(output, f, indent=2)

    def _cluster_phrases(self, phrases, parent_embed):
        """Cluster phrases into topics with improved diversity and relevance.
        
        Args:
            phrases: List of phrases to cluster
            parent_embed: Parent topic embedding for hierarchical context
        """
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        model = model.to(self.device)
        
        # Compute embeddings
        with torch.no_grad():
            feats = model.encode(phrases, convert_to_tensor=True)
            feats = F.normalize(feats, p=2, dim=1)  # Normalize embeddings
            
            # If we have parent embedding, calculate relevance to parent
            if parent_embed is not None:
                # Normalize parent embedding
                parent_embed = F.normalize(parent_embed, p=2, dim=0)
                # Calculate similarity scores [num_phrases]
                parent_sim = torch.matmul(feats, parent_embed)
                # Reshape for broadcasting [num_phrases, 1]
                parent_sim = parent_sim.unsqueeze(1)
                # Apply sigmoid and broadcast across embedding dimension
                feats = feats * F.sigmoid(parent_sim)
            
            feats = feats.cpu().numpy()

        # Adaptive number of clusters based on data size
        actual_num_clusters = min(5, len(phrases) // 5)
        if actual_num_clusters < 2:
            actual_num_clusters = 2

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=actual_num_clusters, random_state=0)
        labels = kmeans.fit(feats).labels_

        # Calculate cluster centers and distances
        centers = kmeans.cluster_centers_
        centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
        
        # Calculate inter-cluster similarity to measure diversity
        inter_sim = np.dot(centers, centers.T)
        np.fill_diagonal(inter_sim, 0)
        diversity_scores = 1 - inter_sim.max(axis=1)
        
        # Calculate phrase-to-center distances
        distances = euclidean_distances(centers, feats)
        relevance_scores = -distances.min(axis=0)  # Negative distance as relevance
        
        # Process each cluster
        subtopics = []
        for cluster_idx in range(actual_num_clusters):
            cluster_mask = (labels == cluster_idx)
            cluster_size = cluster_mask.sum()
            
            if cluster_size < 5:
                continue
            
            # Get cluster phrases and their relevance scores
            cluster_phrases_idx = cluster_mask.nonzero()[0]
            cluster_distances = distances[cluster_idx][cluster_phrases_idx]
            
            # Sort by distance to center
            sorted_idx = cluster_distances.argsort()
            cluster_phrases_idx = cluster_phrases_idx[sorted_idx]
            
            # Select most representative phrase as topic name
            # Prefer longer phrases among the top K most central phrases
            top_k_phrases = [phrases[idx] for idx in cluster_phrases_idx[:5]]
            topic_name = max(top_k_phrases, key=lambda x: len(x.split()))
            
            # Collect phrases with their relevance scores
            topic_phrases = []
            for idx in cluster_phrases_idx:
                phrase = phrases[idx]
                relevance = float(relevance_scores[idx])  # Convert to float for JSON serialization
                topic_phrases.append((phrase, relevance))
            
            # Calculate cluster quality score
            quality_score = float(diversity_scores[cluster_idx] * (cluster_size / len(phrases)))
            
            subtopic = {
                "name": topic_name,
                "phrases": topic_phrases,
                "quality_score": quality_score
            }
            subtopics.append(subtopic)

        return subtopics
