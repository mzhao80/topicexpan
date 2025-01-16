import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_length_histogram():
    # Read the CSV file
    df = pd.read_csv('congress/crec2023.csv')
    
    # Calculate lengths
    text_lengths = df['text'].str.len()
    
    # Create histogram
    plt.figure(figsize=(12, 6))
    sns.histplot(data=text_lengths, bins=50)
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
    
    # Read triples and convert to readable format
    with open('congress/topic_triples.txt', 'r') as f:
        triples = [line.strip().split('\t') for line in f]
    
    # Write readable triples
    with open('congress/topic_triples_readable.txt', 'w') as f:
        for triple in triples:
            if len(triple) == 3:
                topic1, topic2, count = triple
                try:
                    readable_triple = f"{topics[int(topic1)]}\t{topics[int(topic2)]}\t{count}\n"
                    f.write(readable_triple)
                except (KeyError, ValueError) as e:
                    print(f"Warning: Could not process triple {triple}: {e}")

if __name__ == "__main__":
    print("Creating text length histogram...")
    create_length_histogram()
    print("Histogram saved as text_length_distribution.png")
    
    print("\nCreating readable topic triples...")
    create_readable_triples()
    print("Readable triples saved as topic_triples_readable.txt")
