import numpy as np
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import os
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_glove_embeddings(glove_path):
    # First convert GloVe to Word2Vec format
    tmp_path = "temp_word2vec.txt"
    glove2word2vec(glove_path, tmp_path)
    
    # Load the converted embeddings
    model = KeyedVectors.load_word2vec_format(tmp_path, binary=False)
    
    # Clean up temporary file
    os.remove(tmp_path)
    return model

def get_topic_embedding(topic_name, glove_model):
    # Split topic name into words
    words = re.findall(r'[a-z]+', topic_name.lower())
    
    # Get embeddings for each word
    embeddings = []
    for word in words:
        if word in glove_model:
            embeddings.append(glove_model[word])
    
    # Average the embeddings
    if embeddings:
        return np.mean(embeddings, axis=0)
    return np.zeros(300)  # Return zero vector if no words found

def process_speech(speech):
    # Basic preprocessing
    speech = speech.lower()
    speech = re.sub(r'[^\w\s]', ' ', speech)
    return speech

def extract_key_phrases(speech, n=5):
    # Simple extraction of most frequent word combinations
    words = speech.split()
    phrases = []
    for i in range(len(words)-1):
        if len(words[i]) > 3 and len(words[i+1]) > 3:  # Only consider meaningful words
            phrases.append(f"{words[i]} {words[i+1]}")
    return list(set(phrases))[:n]

def main():
    # Load GloVe embeddings
    print("Loading GloVe embeddings...")
    glove_model = load_glove_embeddings("glove/glove.6B.300d.txt")
    
    # Read topics
    topics = {}
    with open("congress/topics.txt", "r") as f:
        for line in f:
            idx, topic = line.strip().split('\t')
            topics[int(idx)] = topic
    
    # Generate topic embeddings
    print("Generating topic embeddings...")
    with open("congress/topic_feats.txt", "w") as f:
        f.write(f"{len(topics) + 1} 300\n")  # Header: +1 for unknown topic
        # Write unknown topic embedding (zeros)
        unknown_embedding = np.zeros(300)
        f.write(f"{len(topics)} " + " ".join(map(str, unknown_embedding)) + "\n")
        # Write regular topic embeddings
        for idx in range(len(topics)):
            embedding = get_topic_embedding(topics[idx], glove_model)
            f.write(f"{idx} " + " ".join(map(str, embedding)) + "\n")
    
    # Process speeches
    print("Processing speeches...")
    speeches = []
    doc2phrases = []
    
    with open("congress/speeches_114.txt", "r", encoding='iso-8859-1') as f:
        for i, line in enumerate(f):
            speech_id, speech_text = line.strip().split('|', 1)  # Split on first occurrence of |
            speech = process_speech(speech_text)
            speeches.append(speech)
            phrases = extract_key_phrases(speech)
            doc2phrases.append(f"{i}\t{'\t'.join(phrases)}")
    
    # Write corpus and doc2phrases
    with open("congress/corpus.txt", "w") as f:
        for i, speech in enumerate(speeches):
            f.write(f"{i}\t{speech}\n")
    
    with open("congress/doc2phrases.txt", "w") as f:
        f.write("\n".join(doc2phrases))
    
    # Generate topic triples
    print("Generating topic triples...")
    vectorizer = TfidfVectorizer()
    speech_vectors = vectorizer.fit_transform(speeches)
    
    # Create topic vectors using key terms
    topic_terms = {idx: topics[idx].replace("_", " ") for idx in topics}
    topic_vectors = vectorizer.transform(list(topic_terms.values()))
    
    # Calculate document-topic similarities
    similarities = cosine_similarity(speech_vectors, topic_vectors)
    
    # Generate triples (doc_id, topic_id, weight)
    triples = []
    for doc_id in range(len(speeches)):
        # Get top 3 most relevant topics for each document
        top_topics = np.argsort(similarities[doc_id])[-3:]
        for topic_id in top_topics:
            weight = similarities[doc_id][topic_id]
            if weight > 0.1:  # Only include significant associations
                triples.append(f"{doc_id}\t{topic_id}\t{weight:.3f}")
    
    with open("congress/topic_triples.txt", "w") as f:
        f.write("\n".join(triples))

if __name__ == "__main__":
    main()
