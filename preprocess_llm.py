#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import re
from tqdm import tqdm
from keybert.llm import OpenAI
from keybert import KeyLLM, KeyBERT
from nltk.corpus import stopwords
import nltk
import argparse
import torch
import matplotlib.pyplot as plt
import openai
from sentence_transformers import SentenceTransformer

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

def extract_phrases(docs, keybert_model):
    """Extract meaningful phrases from text using KeyBERT
    
    Args:
        text: raw text string
        keybert_model: KeyBERT model for extraction
        
    Returns:
        list of extracted phrases
    """   
    # Use KeyBERT to extract keyphrases
    return keybert_model.extract_keywords(docs)

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

    client = openai.OpenAI()
    llm = OpenAI(client, model="gpt-4o-mini", chat=True, verbose=True)
    keybert_model = KeyLLM(llm)

    doc2phrases_map = {}
    
    # skip this next section if doc2phrases.txt already exists
    if os.path.exists(os.path.join(args.data_dir, 'doc2phrases.txt')):
        print("doc2phrases.txt already exists, skipping phrase extraction")
        # Load doc2phrases mapping
        with open(os.path.join(args.data_dir, 'doc2phrases.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                doc_id = int(parts[0])
                phrases = parts[1:]
                doc2phrases_map[doc_id] = phrases
    else:
        print("Extracting phrases from documents...")
        # Process each document
        doc2phrases = extract_phrases(valid_speeches['speech'].tolist(), keybert_model)

        # Save doc2phrases.txt
        print("Saving doc2phrases.txt...")
        with open(os.path.join(args.data_dir, 'doc2phrases.txt'), 'w', encoding='utf-8') as f:
            for doc_id in tqdm(range(len(doc2phrases)), 
                            total=len(doc2phrases),
                            desc="Writing phrases"):
                phrases = doc2phrases[doc_id]
                f.write(f"{doc_id}\t{'\t'.join(phrases)}\n")
        
        doc2phrases_map = {doc_id: phrases for doc_id, phrases in enumerate(doc2phrases)}
    

    # Generate topics.txt
    print("Creating topics.txt...")
    topic_idx = 1
    topic_to_topic_idx = {}
    doc_idx_to_topic_idx = []
    
    # Function to get topic from GPT-4
    def get_topic_for_text(text):
        prompt = f"""Given the following congressional speech, provide a single short topic phrase (1-5 words) that best describes its main subject matter. The topic should be specific but not too narrow.

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

    # Process each document to get its topic
    print("Generating topics for documents...")
    for idx, row in tqdm(valid_speeches.iterrows(), 
                        total=len(valid_speeches),
                        desc="Generating topics"):
        topic = get_topic_for_text(row['speech'])
        
        # Add new topic if we haven't seen it before
        if topic not in topic_to_topic_idx:
            topic_to_topic_idx[topic] = topic_idx
            topic_idx += 1
            
        doc_idx_to_topic_idx.append(topic_to_topic_idx[topic])
    
    print("Saving topics.txt...")
    with open(os.path.join(args.data_dir, 'topics.txt'), 'w', encoding='utf-8') as f:
        # Write politics as topic 0
        f.write(f"0\tpolitics\n")
        # Write other topics
        for topic, idx in sorted(topic_to_topic_idx.items(), key=lambda x: x[1]):
            f.write(f"{idx}\t{topic}\n")
    
    # Save topic_triples.txt using generated topics
    print("Creating topic triples...")
    with open(os.path.join(args.data_dir, 'topic_triples.txt'), 'w', encoding='utf-8') as f:
        for doc_idx, row in tqdm(valid_speeches.iterrows(), 
                                total=len(valid_speeches),
                                desc="Creating topic triples"):
                topic_idx = doc_idx_to_topic_idx[doc_idx]
                for phrase_idx, phrase in enumerate(doc2phrases_map[doc_idx]):
                    f.write(f"{doc_idx}\t{topic_idx}\t{phrase_idx}\n")
    
    # Create topic features using BERT embeddings
    print("Creating topic features...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    
    def get_topic_embedding(topic, model):
        # Create a richer context for the topic
        context = f"This is a discussion about {topic} in the context of United States congressional policy and legislation."
        
        # Get BERT embedding using SentenceTransformer
        with torch.cuda.amp.autocast():  # Use mixed precision for faster computation
            embedding = model.encode(context, convert_to_tensor=True, device=device)
        return embedding.cpu().numpy()

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
            topic_features[str(idx)] = embedding.cpu().numpy()
    
    # Add politics (topic 0)
    topic_features["0"] = get_topic_embedding("politics", model)
    
    # Save topic features
    print("Saving topic features...")
    vector_size = len(next(iter(topic_features.values())))
    with open(os.path.join(args.data_dir, 'topic_feats.txt'), 'w', encoding='utf-8') as fout:
        # Write header
        fout.write(f"{len(topic_features) + 1} {vector_size}\n")
        # Write each topic vector
        for topic_id in range(len(topic_features)):
            vector = topic_features[str(topic_id)]
            vector_str = ' '.join(map(str, vector))
            fout.write(f"{topic_id} {vector_str}\n")
        # Write unknown topic vector
        unknown_vector = ' '.join(['0.0'] * vector_size)
        fout.write(f"unknown {unknown_vector}\n")

    print(f"Preprocessing complete! Files have been created in the {args.data_dir}/ directory.")

if __name__ == "__main__":
    main()
