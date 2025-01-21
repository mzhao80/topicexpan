import numpy as np
import torch
import torch.nn.functional as F
from base import BaseTrainer
from utils import MetricTracker
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
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

        # Initialize sentence transformer for embeddings
        self.sentence_transformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.sentence_transformer.to(self.device)

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
                            generated_phrases = self._generate_phrases(parent_embed)
                            print(f"Generated {len(generated_phrases)} phrases for {child_name}")
                            
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

    def _generate_phrases(self, topic_embedding, num_phrases=20):
        """Generate phrases for a given topic embedding."""
        # Ensure topic embedding is properly shaped
        if len(topic_embedding.shape) == 1:
            topic_embedding = topic_embedding.unsqueeze(0)
            
        # Get document embeddings using MiniLM
        doc_texts = [doc[:512] for doc in self.dataset.docs]  # Truncate long docs
        doc_embeddings = self.sentence_transformer.encode(doc_texts, convert_to_tensor=True)
        
        # Calculate topic-document similarities
        sim_scores = F.cosine_similarity(topic_embedding.unsqueeze(1), doc_embeddings.unsqueeze(0), dim=2)
        sim_scores = F.softmax(sim_scores, dim=1)
        
        # Filter documents by similarity threshold
        threshold = 0.1
        relevant_doc_indices = (sim_scores > threshold).nonzero()[:, 1]
        
        # Generate phrases from relevant documents
        generated_phrases = []
        unique_phrases = set()
        
        for doc_idx in relevant_doc_indices:
            # Get document text and embedding
            doc_text = doc_texts[doc_idx]
            doc_embed = doc_embeddings[doc_idx].unsqueeze(0)
            
            # Generate multiple phrases per document with different temperatures
            for temp in [0.7, 0.8, 0.9]:
                tokens = self.model.phrase_decoder.generate(
                    topic_embedding,
                    temperature=temp,
                    top_p=0.9,
                    repetition_penalty=1.2
                )
                phrase = self.dataset.bert_tokenizer.decode(tokens[0], skip_special_tokens=True).strip()
                
                # Filter phrases
                if (len(phrase) > 3 and 
                    phrase not in unique_phrases and
                    not any(p in phrase for p in ['speaker of the house', 'hakeem jeffries', 'kevin mccarthy']) and
                    self._is_valid_phrase(phrase, doc_text)):
                    
                    unique_phrases.add(phrase)
                    generated_phrases.append({
                        'phrase': phrase,
                        'confidence': float(sim_scores[0, doc_idx]),
                        'embedding': self.sentence_transformer.encode(phrase, convert_to_tensor=True)
                    })
                    
            if len(generated_phrases) >= num_phrases:
                break
                
        return generated_phrases
        
    def _is_valid_phrase(self, phrase, doc_text):
        """Check if phrase is valid."""
        # Convert to lowercase for matching
        phrase = phrase.lower()
        doc_text = doc_text.lower()
        
        # Basic checks
        if len(phrase.split()) < 2:  # Require at least 2 words
            return False
            
        if not all(len(word) > 1 for word in phrase.split()):  # No single-letter words
            return False
            
        # Check if phrase or similar n-grams appear in document
        doc_words = doc_text.split()
        phrase_words = phrase.split()
        n = len(phrase_words)
        
        for i in range(len(doc_words) - n + 1):
            ngram = ' '.join(doc_words[i:i+n])
            if phrase in ngram or ngram in phrase:
                return True
                
        return False
        
    def _cluster_phrases(self, phrases, parent_embedding, num_clusters=None):
        """Cluster generated phrases using hierarchical clustering."""
        if not phrases:
            return []
            
        # Get embeddings and metadata
        embeddings = torch.stack([p['embedding'] for p in phrases]).cpu().numpy()
        phrase_texts = [p['phrase'] for p in phrases]
        confidences = [p['confidence'] for p in phrases]
        
        # Compute topic coherence for different numbers of clusters
        max_clusters = min(5, len(phrases) // 3)
        best_score = -float('inf')
        best_clusters = None
        best_n = None
        
        for n in range(2, max_clusters + 1):
            # Perform hierarchical clustering
            clustering = AgglomerativeClustering(n_clusters=n)
            clusters = clustering.fit_predict(embeddings)
            
            # Group phrases by cluster
            clustered_phrases = [[] for _ in range(n)]
            for phrase, conf, emb, cluster_idx in zip(phrase_texts, confidences, embeddings, clusters):
                clustered_phrases[cluster_idx].append((phrase, conf, emb))
            
            # Calculate topic coherence score
            score = self._calculate_topic_coherence(clustered_phrases)
            
            if score > best_score:
                best_score = score
                best_clusters = clustered_phrases
                best_n = n
                
        # Filter and sort clusters
        valid_clusters = []
        for cluster in best_clusters:
            if len(cluster) >= 3:  # Minimum size threshold
                # Sort phrases within cluster by confidence
                cluster.sort(key=lambda x: x[1], reverse=True)
                valid_clusters.append(cluster)
                
        return valid_clusters
        
    def _calculate_topic_coherence(self, clusters):
        """Calculate topic coherence score for clusters."""
        total_score = 0
        
        for cluster in clusters:
            if len(cluster) < 3:
                continue
                
            # Get phrase embeddings
            cluster_embeddings = np.array([emb for _, _, emb in cluster])
            
            # Calculate average pairwise similarity
            similarities = cosine_similarity(cluster_embeddings)
            np.fill_diagonal(similarities, 0)
            avg_similarity = similarities.mean()
            
            # Calculate distance to parent topic
            parent_dist = np.mean([1 - cos_sim for _, _, emb in cluster])
            
            # Combine metrics
            cluster_score = avg_similarity * (1 - parent_dist)
            total_score += cluster_score
            
        return total_score / len(clusters) if clusters else 0
