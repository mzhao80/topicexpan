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
import matplotlib.pyplot as plt

# Download stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

seed_keywords = [
    "agriculture and food",
    "animals",
    "armed forces and national security",
    "arts, culture, religion",
    "civil rights and liberties, minority issues",
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
    "geographic areas, entities, and committees",
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
    "science, technology, communications",
    "social sciences and history",
    "social welfare",
    "sports and recreation",
    "taxation",
    "transportation and public works",
    "water resources development",
]

def clean_text(text):
    # Replace Madam Speaker, Mr. President, Madam President with Mr. Speaker
    text = re.sub(r'Mr\. President', 'Mr. Speaker', text)
    text = re.sub(r'Mr\. Clerk', 'Mr. Speaker', text)
    text = re.sub(r'Mr\. Chair', 'Mr. Speaker', text)
    text = re.sub(r'Mr\. Chairman', 'Mr. Speaker', text)
    text = re.sub(r'Mr\. Speakerman', 'Mr. Speaker', text)
    text = re.sub(r'Madam President', 'Mr. Speaker', text)
    text = re.sub(r'Madam Speaker', 'Mr. Speaker', text)
    text = re.sub(r'Madam Clerk', 'Mr. Speaker', text)
    text = re.sub(r'Madam Chair', 'Mr. Speaker', text)
    text = re.sub(r'Madam Chairman', 'Mr. Speaker', text)
    text = re.sub(r'Madam Chairwoman', 'Mr. Speaker', text)

    # strip out the following phrases from the beginning of each text and leave the remainder:
    # "Mr. Speaker, " 
    text = re.sub(r'^Mr\. Speaker, ', '', text)
    # "Mr. Speaker, I yield myself the balance of my time. "
    text = re.sub(r'^I yield myself the balance of my time\. ', '', text)
    # "I yield myself such time as I may consume. "
    text = re.sub(r'^I yield myself such time as I may consume\. ', '', text)
    
    return text

def extract_phrases(text, keybert_model):
    """Extract meaningful phrases from text using KeyBERT
    
    Args:
        text: raw text string
        keybert_model: KeyBERT model for extraction
        
    Returns:
        list of extracted phrases
    """   
    # Use KeyBERT to extract keyphrases
    keyphrases = keybert_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1,10),
        stop_words='english',
        use_mmr=True,
        diversity=0.7,
        seed_keywords=seed_keywords
    )
    
    return [phrase for phrase, score in keyphrases if score > 0.3]

def get_bert_embedding(text, model):
    """Get BERT embedding for a piece of text"""
    # KeyBERT's model has a encode method that returns embeddings
    # The encode method automatically uses the same device as the model
    embedding = model.encode([text], convert_to_numpy=True)[0]
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

def create_topic_hierarchy(policy_areas):
    # Create hierarchy using indices
    hierarchy = {
        '0': {'parent': 'root', 'children': [str(i) for i in range(1, len(policy_areas)+1)], 'terms': []}  # Skip politics
    }
    print(hierarchy)
    
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

def plot_similarity_distributions(topic_similarities, phrase_similarities, output_dir):
    """Plot histograms of topic and phrase similarities"""
    plt.figure(figsize=(12, 5))
    
    # Plot topic similarities
    plt.subplot(1, 2, 1)
    plt.hist(topic_similarities, bins=50, edgecolor='black')
    plt.title('Distribution of Best Topic Similarities')
    plt.xlabel('Similarity Score')
    plt.ylabel('Count')
    
    # Plot phrase similarities
    plt.subplot(1, 2, 2)
    plt.hist(phrase_similarities, bins=50, edgecolor='black')
    plt.title('Distribution of Phrase Similarities')
    plt.xlabel('Similarity Score')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'similarity_distributions.png'))
    plt.close()

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
    # apply clean_text to valid_speeeches
    valid_speeches['speech'] = valid_speeches['speech'].apply(clean_text)

    # cut valid speeches to first 1000
    valid_speeches = valid_speeches[:1000]
    
    with open(os.path.join(args.data_dir, 'corpus.txt'), 'w', encoding='utf-8') as f:
        for idx, text in tqdm(enumerate(valid_speeches['speech']), 
                            total=len(valid_speeches), 
                            desc="Writing corpus"):
            f.write(f"{idx}\t{text}\n")
    
    # Initialize KeyBERT with GPU support
    print("Initializing KeyBERT model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    keybert_model = KeyBERT(model=model)
    
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
    hierarchy, policy_areas = create_topic_hierarchy(seed_keywords)
    
    # Save topics.txt
    print("Saving topics.txt...")
    with open(os.path.join(args.data_dir, 'topics.txt'), 'w', encoding='utf-8') as f:
        # Write politics as topic 0
        f.write(f"0\tpolitics\n")
        # Write other policy areas
        for i, area in enumerate(policy_areas):
            f.write(f"{i+1}\t{area}\n")
    
    # Save topic_hier.txt
    print("Saving topic hierarchy...")
    with open(os.path.join(args.data_dir, 'topic_hier.txt'), 'w', encoding='utf-8') as f:
        # Write politics -> policy areas relationships
        for area in hierarchy['0']['children']:
            f.write(f"0\t{area}\n")
    
    # Create topic features using BERT
    print("Creating topic features...")
    topic_features = create_topic_features(["politics"] + policy_areas, model)
    topic_vectors = {str(idx): vec for idx, vec in enumerate(topic_features.values())}
    vector_size = len(next(iter(topic_vectors.values())))  # Get dimension from first vector
    
    # Save topic features
    print("Saving topic features...")
    with open(os.path.join(args.data_dir, 'topic_feats.txt'), 'w', encoding='utf-8') as fout:
        # First write the number of topics (including unknown) and vector dimension
        fout.write(f"{len(topic_vectors) + 1} {vector_size}\n")
        # Write each topic vector
        for topic_id in range(len(topic_vectors)):
            vector = topic_vectors[str(topic_id)]
            vector_str = ' '.join(map(str, vector))
            fout.write(f"{topic_id} {vector_str}\n")
        # Write the unknown topic vector (all zeros) with index len(topic_vectors)
        unknown_vector = ' '.join(['0.0'] * vector_size)
        fout.write(f"unknown {unknown_vector}\n")
    
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
            
            # First find the most relevant topic for this document
            doc_text = row['speech']
            doc_vec = get_bert_embedding(doc_text, model)
            doc_norm = np.linalg.norm(doc_vec)
            
            # Compute similarity with each topic
            topic_sims = []
            for topic_idx, topic in enumerate(policy_areas):
                topic_vec = topic_features[topic]
                topic_norm = np.linalg.norm(topic_vec)
                
                if topic_norm == 0:
                    topic_sims.append(0)
                    continue
                    
                sim = np.dot(doc_vec, topic_vec) / (doc_norm * topic_norm)
                topic_sims.append(sim)
                
            best_topic_idx = np.argmax(topic_sims)
            best_topic_sim = topic_sims[best_topic_idx]
            
            # Skip if the best topic similarity is too low
            if best_topic_sim < 0.3:  # Adjust this threshold as needed
                continue
                
            # Now find the most relevant phrase for this topic
            topic_vec = topic_features[policy_areas[best_topic_idx]]
            topic_norm = np.linalg.norm(topic_vec)
            
            # Compute similarity between each phrase and the best topic
            phrase_sims = []
            for phrase_idx, phrase in enumerate(phrases):
                # Get phrase embedding
                if phrase in embedding_cache:
                    phrase_vec = embedding_cache[phrase]
                else:
                    phrase_vec = get_bert_embedding(phrase, model)
                    embedding_cache[phrase] = phrase_vec
                
                # Compute cosine similarity
                phrase_norm = np.linalg.norm(phrase_vec)
                if phrase_norm == 0:
                    phrase_sims.append((phrase_idx, 0))
                    continue
                    
                sim = np.dot(phrase_vec, topic_vec) / (phrase_norm * topic_norm)
                phrase_sims.append((phrase_idx, sim))
            
            # Sort by similarity score
            phrase_sims.sort(key=lambda x: x[1], reverse=True)
            
            # Get the top phrase that exceeds similarity threshold
            top_phrases = []
            for ph_idx, sim in phrase_sims:
                if sim > 0.3:  # Adjust this threshold as needed
                    top_phrases.append((ph_idx, sim))
            # get top phrase by second index
            if top_phrases:
                top_phrase = max(top_phrases, key = lambda x: x[1])            
                f.write(f"{doc_idx}\t{best_topic_idx+1}\t{top_phrase[0]}\n")
    
    # Plot similarity distributions
    #plot_similarity_distributions(all_topic_sims, all_phrase_sims, args.data_dir)
    
    print(f"Preprocessing complete! Files have been created in the {args.data_dir}/ directory.")
    #print(f"Similarity distribution plots saved as similarity_distributions.png")

if __name__ == "__main__":
    main()
