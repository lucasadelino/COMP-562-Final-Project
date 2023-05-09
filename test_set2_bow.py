"""Test bag of words approach"""

import pickle
import numpy as np
from sklearn.cluster import KMeans
from test import *

# Load pickle files
with open('matrix.pkl', 'rb') as file:
    mat = pickle.load(file)
with open('vocab.pkl', 'rb') as file:
    vocab = pickle.load(file)
with open('test2_descriptors.pkl', 'rb') as file:
    queries = pickle.load(file)

# Calculate accuracy using most similar vector
errors = 0
for i in range(528):
    if queries[i] is None:
        continue
    result, _ = similarity(generate_vector(queries[i], vocab), mat)
    if result != i:
        errors += 1
    print(f'Got {i}')
print(f'Accuracy:{(528 - errors)/528}')

# OPTIONAL: Uncomment to calculate accuracy if true match is in the top n 
# most similar matches. This might indicate whether the algorithm might work
# for image retrieval tasks
"""# Calculate accuracy using top 5 most similar vectors
# TODO: Integrate with the for loop above
for i in range(528):
    if queries[i] is None:
        continue
    result = n_most_similar(5, generate_vector(queries[i], vocab), mat)
    if i not in result:
        errors += 1
    print(f'Got {i}')
print(f'Accuracy:{(528 - errors)/528}')"""
