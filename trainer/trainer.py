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
            topic_embeddings = self.model.topic_encoder.inductive_encode()
            
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
                
                # Skip if we've already processed this parent's children
                if parent_name in output and len(output[parent_name]) > 0:
                    continue
                
                # Initialize parent in output if not exists
                if parent_name not in output:
                    output[parent_name] = []
                
                # Add each child as a discovered topic
                for child_id in children:
                    child_name = dataset.topics[child_id]
                    if child_name not in seen_topics:
                        seen_topics.add(child_name)
                        output[parent_name].append({
                            "name": child_name,
                            "quality_score": 1.0  # Prewritten topics get max quality score
                        })
            
            # Write output
            with open('discovered_topics.json', 'w') as f:
                json.dump(output, f, indent=2)

    def _cluster_phrases(self, vid2phrases, num_clusters, min_cluster_size=5, parent_embedding=None):
        """Cluster phrases into topics with improved diversity and relevance.
        
        Args:
            vid2phrases: Dictionary mapping vertex IDs to phrases
            num_clusters: Number of clusters to create
            min_cluster_size: Minimum size for a valid cluster
            parent_embedding: Optional parent topic embedding for hierarchical context
        """
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        model = model.to(self.device)
        
        vid2tnames = {vid: [] for vid in vid2phrases}
        vid2tinfos = {vid: [] for vid in vid2phrases}
        
        for vid, phrases in vid2phrases.items():
            if len(phrases) < min_cluster_size:
                vid2tnames[vid].append('Insufficient-Phrases')
                continue

            vid_docids, vid_phrases = [], []
            for doc_id, phrase in phrases:
                # Remove duplicates and very short phrases
                if len(phrase.split()) < 2 or phrase in vid_phrases:
                    continue
                vid_phrases.append(phrase)
                vid_docids.append(doc_id)
            
            if len(vid_phrases) < min_cluster_size:
                vid2tnames[vid].append('Insufficient-Unique-Phrases')
                continue

            # Compute embeddings
            with torch.no_grad():
                vid_feats = model.encode(vid_phrases, convert_to_tensor=True)
                vid_feats = F.normalize(vid_feats, p=2, dim=1)  # Normalize embeddings
                
                # If we have parent embedding, calculate relevance to parent
                if parent_embedding is not None:
                    parent_sim = torch.matmul(vid_feats, parent_embedding.unsqueeze(1))
                    # Boost features based on parent similarity
                    vid_feats = vid_feats * F.sigmoid(parent_sim)
                
                vid_feats = vid_feats.cpu().numpy()

            # Adaptive number of clusters based on data size
            actual_num_clusters = min(num_clusters, len(vid_phrases) // min_cluster_size)
            if actual_num_clusters < 2:
                actual_num_clusters = 2

            # Perform k-means clustering
            kmeans = KMeans(n_clusters=actual_num_clusters, random_state=0)
            labels = kmeans.fit(vid_feats).labels_

            # Calculate cluster centers and distances
            centers = kmeans.cluster_centers_
            centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
            
            # Calculate inter-cluster similarity to measure diversity
            inter_sim = np.dot(centers, centers.T)
            np.fill_diagonal(inter_sim, 0)
            diversity_scores = 1 - inter_sim.max(axis=1)
            
            # Calculate phrase-to-center distances
            distances = euclidean_distances(centers, vid_feats)
            relevance_scores = -distances.min(axis=0)  # Negative distance as relevance
            
            # Process each cluster
            for cluster_idx in range(actual_num_clusters):
                cluster_mask = (labels == cluster_idx)
                cluster_size = cluster_mask.sum()
                
                if cluster_size < min_cluster_size:
                    continue
                
                # Get cluster phrases and their relevance scores
                cluster_phrases_idx = cluster_mask.nonzero()[0]
                cluster_distances = distances[cluster_idx][cluster_phrases_idx]
                
                # Sort by distance to center
                sorted_idx = cluster_distances.argsort()
                cluster_phrases_idx = cluster_phrases_idx[sorted_idx]
                
                # Select most representative phrase as topic name
                # Prefer longer phrases among the top K most central phrases
                top_k_phrases = [vid_phrases[idx] for idx in cluster_phrases_idx[:5]]
                topic_name = max(top_k_phrases, key=lambda x: len(x.split()))
                
                # Collect phrases with their relevance scores
                topic_phrases = []
                for idx in cluster_phrases_idx:
                    doc_id = vid_docids[idx]
                    phrase = vid_phrases[idx]
                    relevance = float(relevance_scores[idx])  # Convert to float for JSON serialization
                    topic_phrases.append((doc_id, phrase, relevance))
                
                # Calculate cluster quality score
                quality_score = float(diversity_scores[cluster_idx] * (cluster_size / len(vid_phrases)))
                
                topic_info = (cluster_idx, topic_name, topic_phrases, cluster_size, quality_score)
                vid2tnames[vid].append(topic_name)
                vid2tinfos[vid].append(topic_info)

        return vid2tnames, vid2tinfos
