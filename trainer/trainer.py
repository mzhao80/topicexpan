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

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, batch_data in enumerate(self.data_loader):
            doc_ids, doc_infos, topic_ids, phrase_infos = batch_data
            
            # Debug: Check input shapes and values
            if batch_idx == 0:  # Only print for first batch
                print("\n[DEBUG] Batch Information:")
                print(f"Document IDs shape: {doc_ids.shape}")
                print(f"Topic IDs shape: {topic_ids.shape}")
                print(f"Unique topics in batch: {topic_ids.unique().tolist()}")
                print(f"Input sequence lengths: {doc_infos['attention_mask'].sum(1).tolist()}")
                print(f"Target phrase lengths: {phrase_infos['attention_mask'].sum(1).tolist()}")
            
            encoder_input = {k: v.to(self.device) for k, v in doc_infos.items()}
            decoder_input = {k: v[:, :-1].to(self.device) for k, v in phrase_infos.items()}
            
            sim_target = topic_ids.to(self.device)
            gen_target = phrase_infos['input_ids'][:, 1:].to(self.device)

            self.optimizer.zero_grad()
            
            sim_score, gen_score = self.model(encoder_input, decoder_input, topic_ids)
            
            # Debug: Check model outputs
            if batch_idx == 0:  # Only print for first batch
                print("\n[DEBUG] Model Outputs:")
                print(f"Similarity scores shape: {sim_score.shape}")
                print(f"Score range: [{sim_score.min():.3f}, {sim_score.max():.3f}]")
                print(f"Generation logits shape: {gen_score.shape}")
                
                # Print predicted vs actual topics
                pred_topics = sim_score.argmax(dim=1)
                print("\nTopic Prediction Sample:")
                for i in range(min(3, len(pred_topics))):
                    print(f"Pred: {pred_topics[i].item()} | True: {sim_target[i].item()}")
                
                # Print sample phrase
                if hasattr(self.dataset, 'bert_tokenizer'):
                    gen_tokens = gen_score[0].argmax(dim=1)
                    generated = self.dataset.bert_tokenizer.decode(gen_tokens)
                    target = self.dataset.bert_tokenizer.decode(gen_target[0])
                    print("\nPhrase Generation Sample:")
                    print(f"Generated: {generated}")
                    print(f"Target: {target}")
            
            sim_loss = self.criterions['sim'](sim_score, sim_target)
            gen_loss = self.criterions['gen'](gen_score, gen_target)
            loss = sim_loss + gen_loss
            
            # Debug: Check loss values
            if batch_idx % 100 == 0:
                print(f"\n[DEBUG] Batch {batch_idx} Losses:")
                print(f"Similarity Loss: {sim_loss.item():.3f}")
                print(f"Generation Loss: {gen_loss.item():.3f}")
                print(f"Total Loss: {loss.item():.3f}")

            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('sim_loss', sim_loss.item())
            self.train_metrics.update('gen_loss', gen_loss.item())

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        if self.do_validation:
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            self.logger.info(f'[{current_time}] Starting validation for epoch: {epoch}')
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
                gen_target = phrase_infos['input_ids'][:, 1:].to(self.device)

                sim_score, gen_score = self.model(encoder_input, decoder_input, topic_ids)
                sim_loss = self.criterions['sim'](sim_score, sim_target)
                gen_loss = self.criterions['gen'](gen_score, gen_target)
                loss = sim_loss + gen_loss

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('sim_loss', sim_loss.item()) 
                self.valid_metrics.update('gen_loss', gen_loss.item())
                self.valid_metrics.update('loss', loss.item())      
                
                # output and target are of shape (batch_size, num_classes)
                for met in self.metric_ftns:
                    if len(gen_target) == 0: continue
                    if met.__name__ == 'embedding_sim':
                        gen_output = self.model.gen(encoder_input, topic_ids)
                        # Get attention masks
                        output_mask = (gen_output != self.dataset.bert_tokenizer.pad_token_id).float()
                        target_mask = (gen_target != self.dataset.bert_tokenizer.pad_token_id).float()
                        met_val = met(gen_output, gen_target, output_mask, target_mask)
                    else:
                        met_val = met(gen_score, gen_target)
                    self.valid_metrics.update(met.__name__, met_val)

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Validation Epoch: {} {} Loss: {:.6f}'.format(
                        epoch,
                        self._progress_validation(batch_idx),
                        loss.item()))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
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


    def infer(self, data_loader):
        self.model.eval()
        
        # Get topic embeddings from GCN
        topic_embeddings = self.model.topic_encoder.inductive_encode()
        
        # Initialize the hierarchy with prewritten topics
        discovered_topics = {}
        for parent_id, children in data_loader.dataset.topic_hier.items():
            if len(children) > 0:  # Only process parents that have children
                parent_name = data_loader.dataset.topics[data_loader.dataset.topicRank2topicID[parent_id]]
                discovered_topics[parent_name] = {
                    "children": [],
                    "quality_score": 1.0  # Prewritten topics get max quality score
                }
                for child_id in children:
                    child_name = data_loader.dataset.topics[data_loader.dataset.topicRank2topicID[child_id]]
                    discovered_topics[parent_name]["children"].append({
                        "name": child_name,
                        "quality_score": 1.0  # Prewritten topics get max quality score
                    })

        # Now expand each parent topic
        for parent_id, parent_embed in topic_embeddings.items():
            parent_name = data_loader.dataset.topics[data_loader.dataset.topicRank2topicID[parent_id]]
            
            # Skip if parent already has children from prewritten hierarchy
            if parent_name in discovered_topics and len(discovered_topics[parent_name]["children"]) > 0:
                continue
                
            # Generate phrases for this parent topic
            parent_embed = parent_embed.unsqueeze(0)  # Add batch dimension
            generated_phrases = self.model.phrase_decoder.generate(parent_embed)
            generated_phrases = self.bert_tokenizer.batch_decode(generated_phrases, skip_special_tokens=True)
            
            # Cluster phrases into subtopics
            subtopics = self._cluster_phrases(generated_phrases, parent_embed)
            
            # Store discovered topics
            if parent_name not in discovered_topics:
                discovered_topics[parent_name] = {
                    "children": [],
                    "quality_score": 0.8  # Default quality score for discovered topics
                }
            
            for subtopic in subtopics:
                discovered_topics[parent_name]["children"].append({
                    "name": subtopic["name"],
                    "quality_score": subtopic["quality_score"]
                })
        
        return discovered_topics

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
                parent_sim = torch.matmul(feats, parent_embed)
                # Boost features based on parent similarity
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
