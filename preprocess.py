#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import re
from collections import defaultdict
import json
from gensim.models import KeyedVectors
from tqdm import tqdm
from keybert import KeyBERT
from nltk.corpus import stopwords
import nltk
import argparse
import torch

# Download stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def clean_text(text):
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', str(text))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_phrases(text, keybert_model):
    """Extract meaningful phrases from text using KeyBERT
    
    Args:
        text: raw text string
        keybert_model: KeyBERT model for extraction
        
    Returns:
        list of extracted phrases
    """
    # Clean the text first
    text = clean_text(text)
    
    # Use KeyBERT to extract keyphrases
    keyphrases = keybert_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 4),
        stop_words='english',
        use_maxsum=True,
        nr_candidates=20,
        top_n=5
    )
    
    # Extract just the phrases (without scores)
    phrases = [phrase for phrase, score in keyphrases]
    
    return phrases

def get_bert_embedding(text, model):
    """Get BERT embedding for a piece of text"""
    # KeyBERT's model has a encode method that returns embeddings
    # The encode method automatically uses the same device as the model
    embedding = model.model.encode([text], convert_to_numpy=True)[0]
    return embedding

def create_topic_features(topics, model):
    """Create feature vectors for topics using BERT embeddings
    
    For multi-word topics:
    1. Replace hyphens/underscores with spaces
    2. Get BERT embedding for the entire phrase
    """
    features = {}
    
    for topic in topics:
        # Clean the topic name
        cleaned_topic = topic.replace('_', ' ').replace('-', ' ')
        # Get BERT embedding
        features[topic] = get_bert_embedding(cleaned_topic, model)
    
    return features

def create_topic_hierarchy():
    policy_areas = [
        "politics",
        "agriculture_and_food",
        "animals",
        "armed_forces_and_national_security",
        "arts_culture_religion",
        "civil_rights_and_liberties_minority_issues",
        "commerce",
        "crime_and_law_enforcement",
        "economics_and_public_finance",
        "education",
        "emergency_management",
        "energy",
        "environmental_protection",
        "families",
        "finance_and_financial_sector",
        "foreign_trade_and_international_finance",
        "geographic_areas_entities_and_committees",
        "government_operations_and_politics",
        "health",
        "housing_and_community_development",
        "immigration",
        "international_affairs",
        "labor_and_employment",
        "law",
        "native_americans",
        "private_legislation",
        "public_lands_and_natural_resources",
        "science_technology_communications",
        "social_sciences_and_history",
        "social_welfare",
        "sports_and_recreation",
        "taxation",
        "transportation_and_public_works",
        "water_resources_development",
    ]
    
    # Create hierarchy using indices
    hierarchy = {
        'root': {'children': ['0'], 'terms': []},  # 0 is politics
        '0': {'parent': 'root', 'children': [str(i) for i in range(1, len(policy_areas))], 'terms': []}  # Skip politics
    }
    
    # Add each policy area to hierarchy using indices
    for i in range(1, len(policy_areas)):  # Skip politics
        hierarchy[str(i)] = {
            'parent': '0',  # Parent is politics (index 0)
            'children': [],
            'terms': []
        }
    
    return hierarchy, policy_areas

stop_words = set(stopwords.words('english'))

def compute_topic_similarity(doc_text, topic_vec, model):
    """Compute similarity between a document and a topic"""
    # Get document embedding
    doc_vec = get_bert_embedding(doc_text, model)
    
    # Compute cosine similarity
    doc_norm = np.linalg.norm(doc_vec)
    topic_norm = np.linalg.norm(topic_vec)
    
    if doc_norm == 0 or topic_norm == 0:
        return 0.0
        
    similarity = np.dot(doc_vec, topic_vec) / (doc_norm * topic_norm)
    return max(0, similarity)  # Ensure non-negative

def save_vectors_word2vec_format(fname, vectors, vector_size):
    """Save vectors in word2vec format"""
    with open(fname, 'w', encoding='utf-8') as fout:
        # Write header: number of vectors and vector size
        fout.write(f"{len(vectors)} {vector_size}\n")
        # Write vectors
        for word, vector in vectors.items():
            vector_str = ' '.join(map(str, vector))
            fout.write(f"{word} {vector_str}\n")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Preprocess data for topic expansion')
    parser.add_argument('--data-dir', type=str, default='congress',
                      help='Directory to store processed files (default: congress)')
    parser.add_argument('--input-file', type=str, default='congress/crec2023.csv',
                      help='Input CSV file path (default: congress/crec2023.csv)')
    parser.add_argument('--min-words', type=int, default=50,
                      help='Minimum number of words required in a speech (default: 50)')
    args = parser.parse_args()
    
    print("Loading and preprocessing data...")
    
    # Load the CSV file
    df = pd.read_csv(args.input_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Create corpus.txt
    print("Creating corpus.txt...")
    # Filter out short speeches
    valid_speeches = df[
        df['speech'].str.split().str.len() >= args.min_words
    ]

    # only take the first 1000
    valid_speeches = valid_speeches.head(1000)
    
    with open(os.path.join(args.data_dir, 'corpus.txt'), 'w', encoding='utf-8') as f:
        for idx, text in tqdm(enumerate(valid_speeches['speech']), 
                            total=len(valid_speeches), 
                            desc="Writing corpus"):
            f.write(f"{idx}\t{text}\n")
    
    # Initialize KeyBERT with GPU support
    print("Initializing KeyBERT model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    keybert_model = KeyBERT(model='all-MiniLM-L6-v2')  # Specify a smaller but efficient model
    # Move model to GPU
    keybert_model.model.to(device)
    
    # skip this next section if doc2phrases.txt already exists
    if os.path.exists(os.path.join(args.data_dir, 'doc2phrases.txt')):
        print("doc2phrases.txt already exists, skipping phrase extraction")
    else:
        print("Extracting phrases from documents...")
        doc2phrases = {}
        
        # Process each document
        for idx, text in tqdm(enumerate(valid_speeches['speech']), 
                            total=len(valid_speeches),
                            desc="Extracting phrases"):
            phrases = extract_phrases(text, keybert_model)
            doc2phrases[idx] = phrases
        
        # Save doc2phrases.txt
        print("Saving doc2phrases.txt...")
        with open(os.path.join(args.data_dir, 'doc2phrases.txt'), 'w', encoding='utf-8') as f:
            for doc_id in tqdm(sorted(doc2phrases.keys()), 
                            total=len(doc2phrases),
                            desc="Writing phrases"):
                phrases = doc2phrases[doc_id]
                if phrases:  # Only write if there are phrases
                    f.write(f"{doc_id}\t{'\t'.join(phrases)}\n")
    
    # Create topic hierarchy and get policy areas list
    print("Creating topic hierarchy...")
    hierarchy, policy_areas = create_topic_hierarchy()
    
    # Save topics.txt
    print("Saving topics.txt...")
    with open(os.path.join(args.data_dir, 'topics.txt'), 'w', encoding='utf-8') as f:
        for idx, topic in enumerate(policy_areas):
            f.write(f"{idx}\t{topic}\n")
    
    # Save topic_hier.txt
    print("Saving topic hierarchy...")
    with open(os.path.join(args.data_dir, 'topic_hier.txt'), 'w', encoding='utf-8') as f:
        # Write politics -> policy areas relationships
        for area in hierarchy['0']['children']:
            f.write(f"0\t{area}\n")
        # Write policy areas -> subtopics relationships
        for area, info in hierarchy.items():
            if area != '0':  # Skip the root node
                for subtopic in info['children']:
                    f.write(f"{area}\t{subtopic}\n")
    
    # Create topic features using BERT
    print("Creating topic features...")
    topic_features = create_topic_features(policy_areas, keybert_model)
    
    # Save topic_triples.txt using policy areas from the CSV
    print("Creating topic triples...")
    embedding_cache = {}
    
    # Load doc2phrases mapping first
    doc2phrases_map = {}
    with open(os.path.join(args.data_dir, 'doc2phrases.txt'), 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            doc_id = int(parts[0])
            phrases = parts[1:]
            doc2phrases_map[doc_id] = phrases
    
    with open(os.path.join(args.data_dir, 'topic_triples.txt'), 'w', encoding='utf-8') as f:
        for doc_idx, row in tqdm(valid_speeches.iterrows(), 
                                total=len(valid_speeches),
                                desc="Computing document-topic similarities"):
            if doc_idx not in doc2phrases_map:
                continue
                
            phrases = doc2phrases_map[doc_idx]
            if not phrases:
                continue
            
            # For each topic
            for topic_idx, topic in enumerate(policy_areas):
                topic_vec = topic_features[topic]
                topic_norm = np.linalg.norm(topic_vec)
                        
                # Get phrase embedding
                if phrase in embedding_cache:
                    phrase_vec = embedding_cache[phrase]
                else:
                    phrase_vec = get_bert_embedding(phrase, keybert_model)
                    embedding_cache[phrase] = phrase_vec
                
                # Compute cosine similarity
                phrase_norm = np.linalg.norm(phrase_vec)
                if phrase_norm == 0 or topic_norm == 0:
                    phrase_sim = 0.0
                else:
                    phrase_sim = np.dot(phrase_vec, topic_vec) / (phrase_norm * topic_norm)
                phrase_sims.append((ph_idx, phrase_sim))
                phrase_cache[phrase] = phrase_sim
                
                # Write the most relevant phrase for this topic
                if phrase_sims:
                    best_phrase_idx, score = max(phrase_sims, key=lambda x: x[1])
                    if score > 0:
                        f.write(f"{doc_idx}\t{topic_idx}\t{best_phrase_idx}\n")
    
    # Save topic_feats.txt in word2vec format
    print("Saving topic features...")
    topic_vectors = {str(idx): vec for idx, vec in enumerate(topic_features.values())}
    vector_size = len(next(iter(topic_vectors.values())))  # Get dimension from first vector
    topic_vectors['unknown'] = np.zeros(vector_size)
    save_vectors_word2vec_format(os.path.join(args.data_dir, 'topic_feats.txt'), topic_vectors, vector_size)
    
    print(f"Preprocessing complete! Files have been created in the {args.data_dir}/ directory.")

if __name__ == "__main__":
    main()
