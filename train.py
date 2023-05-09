"""Create a feature matrix from all reference images, using a bag-of-visual-
words approach.
Each feature consists of a tf-idf-weighted vector of counts of visual words.
Visual words are generated using k-means clustering: each cluser is represents
a visual word. 
"""

from sklearn.cluster import KMeans
import pickle
import numpy as np

# Helper lambda function for calculating inverse document frequency
idf = lambda v : np.log(528/(np.sum(v > 0, axis=0)))
# Number of clusters
K = 3000

# Helper functions:
def decode(observations:np.ndarray, kmeans) -> np.ndarray:
    """Given a series of observations and a fitted kmeans object, decode each
    observation. 
    This essentially transforms a descriptor matrix into a vector
    of visual words"""    
    words = []
    for observation in observations:
        word = kmeans.predict(observation)
        words.append(word)
    return words

def generate_feature_matrix(vwords: np.ndarray, k:int) -> np.ndarray:
    """Given a matrix of visual words, returns a matrix of counts of each 
    visual word. Each row in each matrix corresponds to one of the reference
    images.
    The resulting matrix is weighted using term frequency-inverse document 
    frequency (TF-IDF), in order to prioritize the most relevant visual words.
    """
    vectors = []
    for image in vwords:
        # Initialize empty vector with one element per word in vocab
        vector = np.zeros(k)
        # Count how often each visual word appears in image 
        for vword in image:
            vector[vword] += 1
        vectors.append(vector)
    
    # Convert to np array
    vectors = np.stack(vectors)
    # Save idf for convenience
    with open('idf.pkl', 'wb') as file:
        pickle.dump(idf(vectors), file)

    # Return weighted vectors
    return vectors * idf(vectors)

# Load reference picture descriptors 
with open('reference_descriptors.pkl', 'rb') as file:
    features = pickle.load(file)

# Convert descriptor from list of vectors to one big matrix containing all vectors
all_features = []
for image in features:
    for descriptor in image:
        all_features.append(descriptor)
# Convert to np array
all_features = np.stack(all_features)

# Generate vocabulary
print('Beginning clustering...')
kmeans = KMeans(n_clusters = K, n_init='auto')
kmeans.fit(all_features)
print('Finished clustering!')

# Decode descriptors and create feature matrix 
matrix = generate_feature_matrix(decode(features, kmeans),K)

# Save feature matrix and vocab for convenience
with open('matrix.pkl', 'wb') as file:
    pickle.dump(matrix, file)
with open('vocab.pkl', 'wb') as file3:
    pickle.dump(kmeans, file3)
