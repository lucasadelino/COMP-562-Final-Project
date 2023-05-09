"""Test bag of words approach for test set 1"""

import pickle
import numpy as np
from sklearn.cluster import KMeans
from test import *

# Load pickle files
with open('matrix.pkl', 'rb') as file:
    mat = pickle.load(file)
with open('vocab.pkl', 'rb') as file:
    vocab = pickle.load(file)
with open('test1_descriptors.pkl', 'rb') as file:
    queries = pickle.load(file)

errors = 0
# Calculate accuracy using most similar vector
for i in range(50):
    if queries[i] is None:
        continue
    result, _ = similarity(generate_vector(queries[i], vocab), mat[:50])
    if result != i:
        errors += 1
    print(f'Got {i}')
print(f'Accuracy:{(50 - errors)/50}')