"""Extract features from query and reference images"""

import csv, cv2, pickle, os
import numpy as np

sift = cv2.SIFT_create()

def extract(image: cv2.Mat, sift) -> np.ndarray:
    """Returns an array of descriptors, given a cv2 image object and a SIFT
    object"""
    _, descriptors = sift.detectAndCompute(image, None)
    return descriptors

test2 = []
references = []
# Use ground truth .csv file to detect which images to extract
with open('ground_truth.csv') as file:
    reader = csv.reader(file)
    for i, row in enumerate(reader):
        # Get and extract reference and query images
        reference = cv2.imread(f'images/reference/{row[1]}.jpg')
        test2_img = cv2.imread(f'images/test2/{row[0]}.jpg')
        test2.append(extract(test2_img, sift))
        references.append(extract(reference, sift))
        print(f'Row {i}: {row[0]}, {row[1]}') # Print out progress
print('Got all rows')

files = os.listdir('./images/test1')

test1 = []
for file in files:
    test1_img = cv2.imread(f'images/test1/{file}')
    test1.append(extract(test1_img, sift))

# Save descriptor matrices for convenience
with open('reference_descriptors.pkl', 'wb') as file:
    pickle.dump(references, file)
with open('test1_descriptors.pkl', 'wb') as file:
    pickle.dump(test1, file)
with open('test2_descriptors.pkl', 'wb') as file:
    pickle.dump(test2, file)
