#!/usr/bin/env python3
import pandas as pd
import numpy as np
import spacy
import os
from collections import defaultdict
import json
from gensim.models import KeyedVectors
from tqdm import tqdm
import re
from keybert import KeyBERT

def clean_text(text):
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', str(text))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_phrases(doc, nlp, keybert_model=None):
    """Extract meaningful phrases from text using both spaCy and KeyBERT
    
    Args:
        doc: spaCy Doc object
        nlp: spaCy model
        keybert_model: Optional KeyBERT model for semantic extraction
        
    Returns:
        list of extracted phrases
    """
    phrases = set()  # Use set to avoid duplicates
    
    # 1. Extract noun phrases using spaCy
    for chunk in doc.noun_chunks:
        phrase = ' '.join([token.text.lower() for token in chunk 
                         if not token.is_stop and not token.is_punct])
        if phrase and 1 <= len(phrase.split()) <= 4:
            phrases.add(phrase)
    
    # 2. Extract named entities
    for ent in doc.ents:
        if ent.label_ in ['ORG', 'PERSON', 'GPE', 'LAW', 'NORP']:  # Organizations, People, Locations, Laws, Political groups
            phrase = ent.text.lower()
            if 1 <= len(phrase.split()) <= 4:
                phrases.add(phrase)
    
    # 3. Extract verb phrases with their objects
    for token in doc:
        if token.pos_ == "VERB":
            # Get the verb
            verb = token.text.lower()
            # Get object of verb if available
            obj_tokens = []
            for child in token.children:
                if child.dep_ in ['dobj', 'pobj'] and not child.is_stop:
                    obj_tokens.extend([t.text.lower() for t in child.subtree 
                                    if not t.is_stop and not t.is_punct])
            if obj_tokens:
                phrase = f"{verb} {' '.join(obj_tokens)}"
                if len(phrase.split()) <= 4:
                    phrases.add(phrase)
    
    # 4. Extract policy-relevant adjective-noun pairs
    for token in doc:
        if token.pos_ == "NOUN" and not token.is_stop:
            # Get adjectives describing this noun
            adj_tokens = [child.text.lower() for child in token.children 
                         if child.pos_ == "ADJ" and not child.is_stop]
            if adj_tokens:
                phrase = f"{' '.join(adj_tokens)} {token.text.lower()}"
                if len(phrase.split()) <= 4:
                    phrases.add(phrase)
    
    # 5. Use KeyBERT for semantic key phrases if available
    if keybert_model is not None:
        text = doc.text
        keywords = keybert_model.extract_keywords(text, 
                                                keyphrase_ngram_range=(1, 4),
                                                stop_words='english',
                                                use_maxsum=True,
                                                nr_candidates=10,
                                                top_n=5)
        for kw, score in keywords:
            if score > 0.3:  # Only keep phrases with good similarity
                phrases.add(kw.lower())
    
    return list(phrases)

def create_topic_hierarchy():
    # List of all policy areas
    policy_areas = [
        "politics",  # Add politics as first topic
        "agriculture and food",
        "animals",
        "armed forces and national security",
        "arts culture religion",
        "civil rights and liberties minority issues",
        "commerce",
        "crime and law enforcement",
        "economics and public finance",
        "education",
        "emergency management",
        "energy",
        "environmental protection",
        "families",
        "finance and financial sector",
        "foreign trade and international finance",
        "geographic areas entities and committees",
        "government operations and politics",
        "health",
        "housing and community development",
        "immigration",
        "international affairs",
        "labor and employment",
        "law",
        "native americans",
        "private legislation",
        "public lands and natural resources",
        "science technology communications",
        "social sciences and history",
        "social welfare",
        "sports and recreation",
        "taxation",
        "transportation and public works",
        "water resources development"
    ]
    
    # Create hierarchy
    hierarchy = {
        'root': {'children': ['politics'], 'terms': []},
        'politics': {'parent': 'root', 'children': policy_areas[1:], 'terms': []}  # Skip politics itself
    }
    
    # Add each policy area to hierarchy
    for area in policy_areas[1:]:  # Skip politics
        hierarchy[area] = {
            'parent': 'politics',
            'children': [],
            'terms': []
        }
    
    return hierarchy, policy_areas

def create_topic_features(topics, word2vec_model):
    """Create feature vectors for topics using word embeddings"""
    features = {}
    
    # For each topic, create a feature vector from its words
    for topic in topics:
        words = topic.split()
        vectors = []
        for word in words:
            if word in word2vec_model:
                vectors.append(word2vec_model[word])
        if vectors:
            features[topic] = np.mean(vectors, axis=0)
        else:
            features[topic] = np.zeros(300)  # Default to zero vector if no words found
    
    return features

def compute_topic_similarity(doc_text, topic_vec, word2vec_model):
    """Compute similarity between a document and a topic"""
    # Get word vectors for document words
    doc_words = doc_text.lower().split()
    doc_vecs = []
    for word in doc_words:
        if word in word2vec_model:
            doc_vecs.append(word2vec_model[word])
    
    if not doc_vecs:
        return 0.0
    
    # Average document vectors
    doc_vec = np.mean(doc_vecs, axis=0)
    
    # Compute cosine similarity
    similarity = np.dot(doc_vec, topic_vec) / (np.linalg.norm(doc_vec) * np.linalg.norm(topic_vec))
    return max(0, similarity)  # Ensure non-negative

def main():
    print("Loading and preprocessing data...")
    
    # Load the CSV file
    df = pd.read_csv('congress/crec2023.csv')
    
    # Create corpus.txt
    print("Creating corpus.txt...")
    # Filter out clerk speeches first
    valid_speeches = df[~df['speech'].str.startswith("The clerk")]
    with open('congress/corpus.txt', 'w', encoding='utf-8') as f:
        for idx, text in tqdm(enumerate(valid_speeches['speech']), 
                            total=len(valid_speeches), 
                            desc="Writing corpus"):
            f.write(f"{idx}\t{text}\n")

    # Initialize spaCy and KeyBERT
    print("Initializing NLP models...")
    # Enable GPU if available
    spacy.require_gpu()
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('sentencizer')  # Add sentencizer for better batch processing
    keybert_model = KeyBERT()
    
    # Extract phrases for each document
    print("Extracting phrases...")
    doc2phrases = {}
    batch_size = 32  # Process documents in batches for GPU efficiency
    
    # Create batches of texts
    texts = valid_speeches['speech'].tolist()
    num_batches = (len(texts) + batch_size - 1) // batch_size  # Ceiling division
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    
    for batch_idx, batch in enumerate(tqdm(batches, 
                                         total=num_batches,
                                         desc="Processing documents")):
        # Process batch with spaCy
        docs = list(nlp.pipe(batch))
        
        # Extract phrases for each doc in batch
        for doc_idx, doc in enumerate(docs):
            global_idx = batch_idx * batch_size + doc_idx
            phrases = extract_phrases(doc, nlp, keybert_model)
            if phrases:
                doc2phrases[global_idx] = phrases
    
    # Save doc2phrases.txt
    print("Saving doc2phrases.txt...")
    with open('congress/doc2phrases.txt', 'w', encoding='utf-8') as f:
        for doc_id in tqdm(sorted(doc2phrases.keys()), 
                          total=len(doc2phrases),
                          desc="Writing phrases"):
            f.write(f"{doc_id}\t{' '.join(doc2phrases[doc_id])}\n")
    
    # Create topic hierarchy and get policy areas list
    print("Creating topic hierarchy...")
    hierarchy, policy_areas = create_topic_hierarchy()
    
    # Save topics.txt
    print("Saving topics.txt...")
    with open('congress/topics.txt', 'w', encoding='utf-8') as f:
        for idx, topic in enumerate(policy_areas):
            f.write(f"{idx}\t{topic}\n")
    
    # Save topic_hier.txt
    with open('congress/topic_hier.txt', 'w', encoding='utf-8') as f:
        # Write root -> politics relationship
        f.write(f"root\tpolitics\n")
        # Write politics -> policy areas relationships
        for area in hierarchy['politics']['children']:
            f.write(f"politics\t{area}\n")
    
    # Load word2vec model for creating meaningful feature vectors
    print("Loading GloVe embeddings...")
    word2vec_path = os.path.expanduser("~/Downloads/topicexpan/glove/glove.6B.300d.txt")
    word2vec_model = KeyedVectors.load_word2vec_format(word2vec_path, binary=False)
    
    # Create feature vectors for all topics
    print("Creating topic feature vectors...")
    topic_features = {}
    for topic in tqdm(policy_areas, 
                     total=len(policy_areas),
                     desc="Computing topic features"):
        words = topic.split()
        vectors = []
        for word in words:
            if word in word2vec_model:
                vectors.append(word2vec_model[word])
        if vectors:
            topic_features[topic] = np.mean(vectors, axis=0)
        else:
            topic_features[topic] = np.zeros(300)
    
    # Save topic_triples.txt using policy areas from the CSV
    print("Creating topic triples...")
    with open('congress/topic_triples.txt', 'w', encoding='utf-8') as f:
        for doc_idx, row in tqdm(valid_speeches.iterrows(), 
                                total=len(valid_speeches),
                                desc="Computing document-topic similarities"):
            doc_text = row['speech']
            # For each document, compute similarity with all topics
            similarities = []
            for topic_idx, topic in enumerate(policy_areas):
                sim = compute_topic_similarity(doc_text, topic_features[topic], word2vec_model)
                similarities.append((topic_idx, sim))
            
            # Sort by similarity and write top matches
            similarities.sort(key=lambda x: x[1], reverse=True)
            for topic_idx, sim in similarities[:3]:  # Write top 3 most similar topics
                # Convert similarity to integer confidence score (0-100)
                confidence = int(sim * 100)
                f.write(f"{doc_idx}\t{topic_idx}\t{confidence}\n")
    
    # Save topic_feats.txt
    print("Saving topic features...")
    with open('congress/topic_feats.txt', 'w', encoding='utf-8') as f:
        f.write(f"{len(policy_areas)} 300\n")  # num_topics topics, 300 dimensions
        for topic in tqdm(policy_areas, 
                         total=len(policy_areas),
                         desc="Writing topic features"):
            feature_vector = topic_features[topic]
            f.write(' '.join(map(str, feature_vector)) + '\n')
    
    print("Preprocessing complete! Files have been created in the congress/ directory.")

if __name__ == "__main__":
    main()
