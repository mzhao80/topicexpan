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


    def infer(self, config):
        self.model.eval()
        dataset = self.data_loader.dataset
        
        with torch.no_grad():
            # Step 1. Score collection
            total_docids, total_scores = [], []
            for batch_idx, batch_data in enumerate(self.data_loader):
                doc_ids = batch_data[0].to(self.device)
                encoder_input = {k: v.to(self.device) for k, v in batch_data[1].items()}

                vsim_score = self.model.inductive_sim(encoder_input)
                total_docids.append(doc_ids)
                total_scores.append(vsim_score)

            total_docids = torch.cat(total_docids, dim=0)
            total_scores = torch.cat(total_scores, dim=0)

            if config['filter_type'] == 'rank':
                conf_scores, conf_indices = torch.topk(total_scores, k=config['topk'], dim=0)
                conf_docids = total_docids[conf_indices]
            elif config['filter_type'] == 'nscore':
                vmax_scores = total_scores.max(dim=0, keepdim=True)[0]
                vmin_scores = total_scores.min(dim=0, keepdim=True)[0]
                total_scores = (total_scores - vmin_scores) / (vmax_scores - vmin_scores)

            # Step 2. Score filtering & Phrase generation
            vid2phrases = {vid: [] for vid in self.model.vid2pid}
            for batch_idx, batch_data in enumerate(self.data_loader):
                doc_ids = batch_data[0].to(self.device)

                doc_indices, vtopic_ids = [], []
                for doc_idx, doc_id in enumerate(doc_ids):
                    if config['filter_type'] == 'rank':
                        selection = (conf_docids == int(doc_id)).nonzero()
                        vtopic_ids.append(selection[:, 1])
                        doc_indices += [doc_idx] * selection.shape[0]

                    elif config['filter_type'] == 'nscore':
                        target_idx = int((total_docids == doc_id).nonzero())
                        selection = (total_scores[target_idx, :] > config['tau']).nonzero()[:, 0]
                        vtopic_ids.append(selection)
                        doc_indices += [doc_idx] * selection.shape[0]

                if len(doc_indices) == 0: continue
                vtopic_ids = torch.cat(vtopic_ids)

                encoder_input = {k: v[doc_indices, :].to(self.device) for k, v in batch_data[1].items()}
                vgen_output = self.model.inductive_gen(encoder_input, vtopic_ids)
                vgen_strings = dataset.bert_tokenizer.batch_decode(vgen_output, skip_special_tokens=True)

                for idx, (doc_idx, vtopic_id) in enumerate(zip(doc_indices, vtopic_ids)):
                    vid2phrases[int(vtopic_id)].append((int(doc_ids[doc_idx]), vgen_strings[idx]))
        
            vid2tnames, vid2tinfos = self._cluster_phrases(vid2phrases, config['num_clusters'])
        
            # Step 3. Output discovered topics
            import json
            output = {}
            for vid, pid in self.model.vid2pid.items():
                # Get parent topic path
                tid, path = str(pid), []
                while True:
                    path.append(dataset.topics[tid])
                    if len(dataset.topic_invhier[tid]) == 0:
                        break
                    tid = dataset.topic_invhier[tid][0]
                path.append('root')
                path = ' -> '.join(path[::-1])
                
                # Format discovered topics
                topics = []
                for topic_idx, topic_name, topic_phrases, topic_size in vid2tinfos[vid]:
                    # Sort phrases by relevance score
                    sorted_phrases = sorted(topic_phrases, key=lambda x: x[2])
                    phrases = [{"doc_id": doc_id, "text": phrase} for doc_id, phrase, _ in sorted_phrases]
                    topics.append({
                        "name": topic_name,
                        "size": topic_size,
                        "phrases": phrases
                    })
                
                output[path] = topics
            
            # Write as JSON for easier parsing
            with open('discovered_topics.json', 'w') as f:
                json.dump(output, f, indent=2)
        
    def _cluster_phrases(self, vid2phrases, num_clusters):
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        model = model.to(self.device)
        
        vid2tnames = {vid: [] for vid in vid2phrases}
        vid2tinfos = {vid: [] for vid in vid2phrases}
        for vid, phrases in vid2phrases.items():
            if len(phrases) == 0:
                vid2tnames[vid].append('Phrase-Not-Found')
                continue

            vid_docids, vid_phrases = [], []
            for doc_id, phrase in phrases:
                vid_phrases.append(phrase)
                vid_docids.append(doc_id)
            
            if len(vid_phrases) == 0:
                vid2tnames[vid].append('Phrase-Not-Found')
                continue

            # Compute embeddings using all-MiniLM-L6-v2
            with torch.no_grad():
                vid_feats = model.encode(vid_phrases, convert_to_tensor=True)
                vid_feats = vid_feats.cpu().numpy()

            # Perform k-means clustering
            kmeans = KMeans(n_clusters=num_clusters, random_state=0)
            labels = kmeans.fit(vid_feats).labels_

            topic_relevance = euclidean_distances(kmeans.cluster_centers_, vid_feats).min(axis=0)
            topic_name_idxs = euclidean_distances(kmeans.cluster_centers_, vid_feats).argmin(axis=1)
            for topic_idx, topic_name_idx in enumerate(topic_name_idxs):
                topic_name = vid_phrases[topic_name_idx]
                topic_phrases = [(vid_docids[phrase_idx], vid_phrases[phrase_idx], topic_relevance[phrase_idx]) \
                        for phrase_idx in (labels == topic_idx).nonzero()[0]]
                topic_info = (topic_idx, topic_name, topic_phrases, len(topic_phrases))
                
                vid2tnames[vid].append(topic_name)
                vid2tinfos[vid].append(topic_info)

        return vid2tnames, vid2tinfos
