#!/usr/bin/env python3
from random import seed
import pandas as pd
import numpy as np
import os
import re
from tqdm import tqdm
import argparse
import torch
import openai
from sentence_transformers import SentenceTransformer

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

def create_topic_hierarchy(policy_areas):
    # Create hierarchy using indices
    hierarchy = {
        '0': {'parent': 'root', 'children': [str(i) for i in range(1, len(policy_areas)+1)], 'terms': []}  # Skip politics
    }
    print(hierarchy)
    
    return hierarchy, policy_areas

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

def extract_phrases(docs, topics, client):
    """Extract meaningful phrases from text using llm
    
    Args:
        docs: list of strings
        topics: list of topics, each corresponding to the same-indexed doc
        client: OpenAI client
        
    Returns:
        list of extracted phrases for each document
    """
    keyphrases = []
    batch_size = 5  # Process in small batches to avoid rate limits
    
    for i in tqdm(range(0, len(docs), batch_size), desc="Extracting keyphrases"):
        batch_docs = docs[i:i + batch_size]
        batch_topics = topics[i:i + batch_size]
        batch_phrases = []
        
        for doc, topic in zip(batch_docs, batch_topics):
            # Truncate document if too long
            doc_words = doc.split()
            if len(doc_words) > 500:
                doc = ' '.join(doc_words[:500]) + "..."
            
            prompt = f"""Extract one or more top key phrases from this congressional speech that are most relevant to the topic '{topic}'. 
            
            Speech: {doc}
            
            Output the phrases as a comma-separated list. For example: healthcare reform, medicare expansion, drug pricing
            Phrases: """
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that extracts specific, policy-relevant key phrases from congressional speeches. Each phrase should be 1-5 words long, exacly from the text of the speech, and directly related to the given topic."},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                # Split response into phrases and clean them
                phrases = [phrase.strip().lower() for phrase in 
                          response.choices[0].message.content.strip().split(',')]     
                batch_phrases.append(phrases)
                
            except Exception as e:
                print(f"Error processing document {i}: {str(e)}")
                batch_phrases.append([])  # Add empty list for failed document
                
        keyphrases.extend(batch_phrases)
        
    return keyphrases

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
    # apply clean_text to valid_speeeches
    valid_speeches['speech'] = valid_speeches['speech'].apply(clean_text)

    # cut valid speeches
    valid_speeches = valid_speeches[:50]
    
    with open(os.path.join(args.data_dir, 'corpus.txt'), 'w', encoding='utf-8') as f:
        for idx, text in tqdm(enumerate(valid_speeches['speech']), 
                            total=len(valid_speeches), 
                            desc="Writing corpus"):
            f.write(f"{idx}\t{text}\n")

    # Initialize OpenAI client
    print("Initializing OpenAI client...")
    client = openai.OpenAI()
    
    # First, generate topics for each document
    print("Generating topics for documents...")
    doc_topics = []
    topic_to_topic_idx = {keyword: idx+1 for idx, keyword in enumerate(seed_keywords)}
    topic_to_topic_idx['politics'] = 0
    topic_idx = len(topic_to_topic_idx)
    doc_idx_to_topic_idx = []
    
    def get_topic_for_text(text):
        prompt = f"""Given the following congressional speech, provide a short topic phrase (1-5 words) that best describes the main political issue being discussed. The topic should be specific.

            Speech: {text}

            Topic: """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that identifies the main topic of congressional speeches. Respond with only the topic phrase, nothing else."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip().lower()

    for idx, row in tqdm(valid_speeches.iterrows(), 
                        total=len(valid_speeches),
                        desc="Generating topics"):
        topic = get_topic_for_text(row['speech'])
        doc_topics.append(topic)
        
        # Add new topic if we haven't seen it before
        if topic not in topic_to_topic_idx:
            topic_to_topic_idx[topic] = topic_idx
            topic_idx += 1
            
        doc_idx_to_topic_idx.append(topic_to_topic_idx[topic])
    
    # Save topics.txt
    print("Saving topics.txt...")
    with open(os.path.join(args.data_dir, 'topics.txt'), 'w', encoding='utf-8') as f:
        for topic, idx in sorted(topic_to_topic_idx.items(), key=lambda x: x[1]):
            f.write(f"{idx}\t{topic}\n")
    
    # Now extract phrases using the generated topics
    print("Extracting phrases from documents...")
    doc2phrases = extract_phrases(valid_speeches['speech'].tolist(), doc_topics, client)
    
    # Save doc2phrases.txt
    print("Saving doc2phrases.txt...")
    with open(os.path.join(args.data_dir, 'doc2phrases.txt'), 'w', encoding='utf-8') as f:
        for doc_id, phrases in enumerate(doc2phrases):
            if phrases:  # Only write if we have phrases
                f.write(f"{doc_id}\t{'\t'.join(phrases)}\n")
    
    # Create topic triples using the generated topics and phrases
    print("Creating topic triples...")
    with open(os.path.join(args.data_dir, 'topic_triples.txt'), 'w', encoding='utf-8') as f:
        for doc_idx in range(len(valid_speeches)):
            topic_idx = doc_idx_to_topic_idx[doc_idx]
            for phrase_idx, phrase in enumerate(doc2phrases[doc_idx]):
                f.write(f"{doc_idx}\t{topic_idx}\t{phrase_idx}\n")
    
    # Create topic features using BERT
    print("Creating topic features...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = SentenceTransformer('all-MiniLM-L6-v2').to(device)

    # Get embeddings for all topics
    topic_features = {}
    print("Generating topic embeddings...")
    # Process topics in batches for better GPU utilization
    batch_size = 32
    topics = list(topic_to_topic_idx.items())
    for i in tqdm(range(0, len(topics), batch_size), desc="Getting topic embeddings"):
        batch_topics = topics[i:i + batch_size]
        contexts = [f"This is a discussion about {topic} in the context of United States congressional policy and legislation." 
                   for topic, _ in batch_topics]
        
        # Get embeddings for batch
        with torch.cuda.amp.autocast():
            embeddings = model.encode(contexts, convert_to_tensor=True, device=device)
        
        # Store embeddings
        for (topic, idx), embedding in zip(batch_topics, embeddings):
            topic_features[idx] = embedding.cpu().numpy()
    
    # Save topic features
    print("Saving topic features...")
    vector_size = len(next(iter(topic_features.values())))
    with open(os.path.join(args.data_dir, 'topic_feats.txt'), 'w', encoding='utf-8') as fout:
        # Write header
        fout.write(f"{len(topic_features) + 1} {vector_size}\n")
        # Write each topic vector
        for topic_id in range(len(topic_features)):
            vector = topic_features[topic_id]
            vector_str = ' '.join(map(str, vector))
            fout.write(f"{topic_id} {vector_str}\n")
        # Write unknown topic vector
        unknown_vector = ' '.join(['0.0'] * vector_size)
        fout.write(f"unknown {unknown_vector}\n")

    print(f"Preprocessing complete! Files have been created in the {args.data_dir}/ directory.")

if __name__ == "__main__":
    main()
