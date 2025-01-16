import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_length_histogram():
    # Read the CSV file
    df = pd.read_csv('congress/crec2023.csv')
    
    # Calculate text length in words by splitting on spaces
    text_lengths = df['speech'].str.split().str.len()
    
    # Create histogram
    plt.figure(figsize=(12, 6))
    # have buckets of size 10, until 200
    sns.histplot(data=text_lengths, bins=range(0, 200, 10))
    plt.title('Distribution of Text Lengths in Congressional Records')
    plt.xlabel('Text Length (characters)')
    plt.ylabel('Count')
    
    # Save the plot
    plt.savefig('text_length_distribution.png')
    plt.close()

def create_readable_triples():
    # Read topics
    topics = {}
    with open('congress/topics.txt', 'r') as f:
        for line in f:
            idx, name = line.strip().split('\t')
            topics[int(idx)] = name

    # Read doc2phrases
    doc2phrases = {}
    with open('congress/doc2phrases.txt', 'r') as f:
        for line in f:
            content = line.strip().split('\t')
            doc = content[0]
            doc2phrases[doc] = content[1:]
    
    # Read triples and convert to readable format
    with open('congress/topic_triples.txt', 'r') as f:
        triples = [line.strip().split('\t') for line in f]
    
    # Write readable triples
    with open('congress/topic_triples_readable.txt', 'w') as f:
        for triple in triples:
            doc, topic, phrase = triple
            f.write(f"Document: {doc}\t")
            f.write(f"Topic: {topics[int(topic)]}\t")
            f.write(f"Phrase: {doc2phrases[doc][int(phrase)]}\n")

if __name__ == "__main__":
    print("Creating text length histogram...")
    create_length_histogram()
    print("Histogram saved as text_length_distribution.png")
    
    print("\nCreating readable topic triples...")
    create_readable_triples()
    print("Readable triples saved as topic_triples_readable.txt")
