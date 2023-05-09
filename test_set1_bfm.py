import pickle
from cv2 import BFMatcher

# Open pickle files
with open('reference_descriptors.pkl', 'rb') as file:
    references = pickle.load(file)
with open('test1_descriptors.pkl', 'rb') as file:
    queries = pickle.load(file)

def n_matches(query, ref):
    """Returns the number of matches between query and reference descriptors"""
    bfm = BFMatcher()
    matches = bfm.knnMatch(query, ref, k=2)

    # Consider only close matches
    good = []
    for i, j in matches:
        if i.distance < 0.5 * j.distance:
            good.append([i])

    return len(good)

# Calculate accuracy
errors = 0
for k, query in enumerate(queries):
    # Skip empty query descriptors
    if query is None:
        continue
    print(f'Getting {k}')
    # Number of matches between true query/reference pair
    matches = n_matches(queries[k], references[k])
    # If there are no matches at all between true pair, we have made an error
    if matches == 0:
        errors += 1
        continue
    for l, reference in enumerate(references):
        # Skip empty reference descriptors
        if reference is None:
            continue
        # If there is any other query/reference pair with more matches than
        # the true pair, we have made an error
        if n_matches(query, reference) > matches:
            errors += 1
            break
print(f'Accuracy: {(50 - errors)/50}')