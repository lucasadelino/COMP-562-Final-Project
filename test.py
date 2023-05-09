"""Contains functions for testing"""

import numpy as np
import pickle

# Vocab size
K = 3000

# Load idf
with open('idf.pkl', 'rb') as file:
    idf = pickle.load(file)

def cosine_similarity(vec1, vec2):
    """Computes cosine similarity between two vectors"""
    return np.dot(vec1, vec2.T)/(np.linalg.norm(vec1) * np.linalg.norm(vec2, axis=1))

def similarity(test_featvec, matrix) -> tuple:
    """Tests a vector of features against a matrix of feature vectors. Computes
    cosine similarity between test vector and all vectors in feature matrix.
    Returns a tuple containing the index of the row in matrix with the 
    biggest cosine similarity with test vector, and the cosine similarity"""
    results = cosine_similarity(test_featvec, matrix)
    return (np.argmax(results), np.max(results))

def n_most_similar(n, test_featvec, matrix)->list:
    """Tests a vector of features against a matrix of feature vectors. Computes
    cosine similarity between test vector and all vectors in feature matrix.
    Returns the indices of the top n rows with biggest cosine similarity values
    """
    results = cosine_similarity(test_featvec, matrix)
    results = np.argsort(-results)[:n]
    return results

def generate_vector(descriptors, vocab):
    """Given a vocabulary words, returns a feature vector, consisting of counts
    of each visual word.
    The resulting vector is weighted using term frequency-inverse document 
    frequency (TF-IDF), in order to prioritize the most relevant visual words.
    """
    vwords = vocab.predict(descriptors)
    vector = np.zeros(K)
    for word in vwords:
        vector[word] += 1
    return vector * idf


