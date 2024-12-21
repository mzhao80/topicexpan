from gensim.scripts.glove2word2vec import glove2word2vec

# File paths
glove_input_file = "~/Downloads/topicexpan/glove/glove.6B.300d.txt"
word2vec_output_file = "~/Downloads/topicexpan/glove/glove.6B.300d.word2vec.txt"

# Convert GloVe format to Word2Vec format
glove2word2vec(glove_input_file, word2vec_output_file)
print(f"Conversion complete! Word2Vec file saved to {word2vec_output_file}")