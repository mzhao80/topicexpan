import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def create_length_histogram(directory):
    # Read the CSV file
    df = pd.read_csv(os.path.join(directory, 'crec2023.csv'))
    
    # Calculate text length in words by splitting on spaces
    text_lengths = df['speech'].str.split().str.len()
    
    # Create histogram
    plt.figure(figsize=(12, 6))
    # have buckets of size 10, until 200
    sns.histplot(data=text_lengths, bins=range(0, 200, 10))
    plt.title('Distribution of Text Lengths in Congressional Records')
    plt.xlabel('Text Length (words)')
    plt.ylabel('Count')
    
    # Save the plot
    output_path = os.path.join(directory, 'text_length_distribution.png')
    plt.savefig(output_path)
    plt.close()

def create_readable_triples(directory):
    # Read topics
    topics = {}
    with open(os.path.join(directory, 'topics.txt'), 'r') as f:
        for line in f:
            idx, name = line.strip().split('\t')
            topics[int(idx)] = name

    # Read doc2phrases
    doc2phrases = {}
    with open(os.path.join(directory, 'doc2phrases.txt'), 'r') as f:
        for line in f:
            content = line.strip().split('\t')
            doc = content[0]
            doc2phrases[doc] = content[1:]
    
    # Read triples and convert to readable format
    with open(os.path.join(directory, 'topic_triples.txt'), 'r') as f:
        triples = [line.strip().split('\t') for line in f]
    
    # Write readable triples
    with open(os.path.join(directory, 'topic_triples_readable.txt'), 'w') as f:
        for triple in triples:
            doc, topic, phrase = triple
            f.write(f"Document: {doc}\t")
            f.write(f"Topic: {topics[int(topic)]}\t")
            f.write(f"Phrase: {doc2phrases[doc][int(phrase)]}\n")

def main(args):
    print(f"Analyzing data in directory: {args.data_dir}")
    print("Creating length histogram...")
    create_length_histogram(args.data_dir)
    print("Creating readable triples...")
    create_readable_triples(args.data_dir)
    print("Analysis complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze data and create visualizations')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing the data files (crec2023.csv, topics.txt, etc.)')
    args = parser.parse_args()
    main(args)
