#!/usr/bin/env python3
import pandas as pd
import numpy as np
import spacy
import os
import re
from collections import defaultdict
import json
from gensim.models import KeyedVectors
from tqdm import tqdm
from keybert import KeyBERT
from nltk.corpus import stopwords
import nltk

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

def load_glove_vectors(glove_path):
    """Load GloVe vectors from file"""
    print("Loading GloVe vectors...")
    word2vec = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading GloVe vectors"):
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            word2vec[word] = vector
    return word2vec

def create_topic_hierarchy():
    # List of all policy areas
    policy_areas = [
        "politics",
        "agriculture_and_food",
        "animals",
        "armed_forces_and_national_security",
        "arts_culture_religion",
        "civil_rights_and_liberties_minority_issues_commerce",
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

def create_topic_features(topics, word2vec_model):
    """Create feature vectors for topics using word embeddings
    
    For multi-word topics:
    1. First try the exact phrase with hyphens/underscores replaced by spaces
    2. If not found, try individual words and average their vectors, ignoring stopwords
    3. If no words found, use zero vector
    """
    features = {}
    vector_dim = len(next(iter(word2vec_model.values())))  # Get dimension from first vector
    stop_words = set(stopwords.words('english'))
    
    for topic in topics:
        # First try the whole phrase with hyphens/underscores replaced by spaces
        cleaned_topic = re.sub(r'[_-]', ' ', topic).lower().strip()
        if cleaned_topic in word2vec_model:
            features[topic] = word2vec_model[cleaned_topic]
            continue
            
        # If not found, split into words and try each word
        words = cleaned_topic.split(" ")
        # Filter out stopwords and get vectors
        vectors = []
        for word in words:
            if word not in stop_words and word in word2vec_model:
                vectors.append(word2vec_model[word])
        
        if vectors:
            # Average the word vectors we found
            features[topic] = np.mean(vectors, axis=0)
        else:
            # If no words found, use zero vector
            print(f"Warning: No word vectors found for topic '{topic}', using zero vector")
            features[topic] = np.zeros(vector_dim)
    
    return features

def compute_topic_similarity(doc_text, topic_vec, word2vec_model):
    """Compute similarity between a document and a topic"""
    # Get document vectors for words that exist in the model
    doc_vecs = []
    for word in doc_text.split():
        if word in word2vec_model:
            doc_vecs.append(word2vec_model[word])
    
    if not doc_vecs:
        return 0.0
    
    # Average document vectors
    doc_vec = np.mean(doc_vecs, axis=0)
    
    # Check for zero vectors
    doc_norm = np.linalg.norm(doc_vec)
    topic_norm = np.linalg.norm(topic_vec)
    
    if doc_norm == 0 or topic_norm == 0:
        return 0.0
    
    # Compute cosine similarity
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
    print("Loading and preprocessing data...")
    
    # Load the CSV file
    df = pd.read_csv('congress/crec2023.csv')
    
    # Load config to get GloVe path
    with open('config_files/config_congress.json', 'r') as f:
        config = json.load(f)
    glove_path = os.path.expanduser(os.path.join(config['embed_dir'], 'glove.6B.300d.txt'))
    
    # Create corpus.txt
    print("Creating corpus.txt...")
    # Filter out clerk speeches and NaN values
    valid_speeches = df[df['speech'].notna() & ~df['speech'].astype(str).str.startswith("The clerk")]
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
    
    # skip this next section if congress/doc2phrases.txt already exists
    if os.path.exists('congress/doc2phrases.txt'):
        print("doc2phrases.txt already exists, skipping phrase extraction")
    else:
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
        # Write politics -> policy areas relationships
        for area in hierarchy['0']['children']:
            f.write(f"0\t{area}\n")
    
    # Load GloVe vectors
    print("Loading GloVe embeddings...")
    word2vec_model = load_glove_vectors(glove_path)
    
    # Create feature vectors for all topics
    print("Creating topic feature vectors...")
    topic_features = create_topic_features(policy_areas, word2vec_model)
    
    # Save topic_triples.txt using policy areas from the CSV
    print("Creating topic triples...")
    
    # Load doc2phrases mapping first
    doc2phrases_map = {}
    with open('congress/doc2phrases.txt', 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                doc_id = int(parts[0])
                phrases = parts[1].split()
                doc2phrases_map[doc_id] = phrases

    with open('congress/topic_triples.txt', 'w', encoding='utf-8') as f:
        for doc_idx, row in tqdm(valid_speeches.iterrows(), 
                                total=len(valid_speeches),
                                desc="Computing document-topic similarities"):
            # Skip if document has no phrases
            if doc_idx not in doc2phrases_map:
                continue
                
            doc_text = row['speech']
            # For each document, compute similarity with all topics
            similarities = []
            for topic_idx, topic in enumerate(policy_areas):
                sim = compute_topic_similarity(doc_text, topic_features[topic], word2vec_model)
                similarities.append((topic_idx, sim))
            
            # Sort by similarity and get top matches
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_topics = similarities[:3]  # Get top 3 most similar topics
            
            # For each top topic, find the most relevant phrases
            phrases = doc2phrases_map[doc_idx]
            for topic_idx, sim in top_topics:
                # Get topic vector
                topic_vec = topic_features[policy_areas[topic_idx]]
                
                # Compute similarity between each phrase and the topic
                phrase_sims = []
                for ph_idx, phrase in enumerate(phrases):
                    # Get phrase vector (average of word vectors)
                    words = phrase.split()
                    word_vecs = [word2vec_model[w] for w in words if w in word2vec_model]
                    if word_vecs:
                        phrase_vec = np.mean(word_vecs, axis=0)
                        # Check for zero vectors
                        phrase_norm = np.linalg.norm(phrase_vec)
                        topic_norm = np.linalg.norm(topic_vec)
                        
                        if phrase_norm == 0 or topic_norm == 0:
                            phrase_sim = 0.0
                        else:
                            phrase_sim = np.dot(phrase_vec, topic_vec) / (phrase_norm * topic_norm)
                        phrase_sims.append((ph_idx, max(0, phrase_sim)))
                
                # Write the most relevant phrase for this topic
                if phrase_sims:
                    best_phrase_idx, _ = max(phrase_sims, key=lambda x: x[1])
                    f.write(f"{doc_idx}\t{topic_idx}\t{best_phrase_idx}\n")
    
    # Save topic_feats.txt in word2vec format
    print("Saving topic features...")
    # Convert topic features to word2vec format (using indices as words)
    topic_vectors = {str(idx): vec for idx, vec in enumerate(topic_features.values())}
    # Add unknown vector as all zeros for masking
    topic_vectors['unknown'] = np.zeros(300)
    save_vectors_word2vec_format('congress/topic_feats.txt', topic_vectors, 300)
    
    print("Preprocessing complete! Files have been created in the congress/ directory.")

if __name__ == "__main__":
    main()
