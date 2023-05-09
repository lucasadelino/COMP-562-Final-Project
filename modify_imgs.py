"""Script that modifies images to generate test set #1"""

import cv2
import random
import os
import numpy as np

files = os.listdir('./images/reference_50')

# Set seed for reproducibility
random.seed(3)
# Possible multipliers
multipliers = [0.4, 0.6, 0.8, 1.2, 1.4, 1.6, 2]
# Outputs will be stored here   
images = []

for i, file in enumerate(files):
    img = cv2.imread(f'./images/reference_50/{file}')

    # Stretch/shrink
    fx = random.choice(multipliers)
    fy = random.choice(multipliers)
    img = cv2.resize(img, None, fx = fx, fy = fy)
    
    # Crop 
    # Crop with a 50% chance
    random.choice([True, False])
    # Get maximum possible x and y coordinates in image 
    max_y, max_x = img.shape[:2]
    # Randomly choose a crop area width and height
    crop_width = int(max_x * random.choice(multipliers[:3]))
    crop_height = int(max_y * random.choice(multipliers[:3]))   
    # Get crop x and y according to chosen width and height
    crop_x = max_x - crop_width
    crop_y = max_y - crop_height
    # Pick random x and y for cropping
    x = random.randint(0, crop_x)
    y = random.randint(0, crop_y)
    # Crop from slice  
    img = img[y:(y+crop_height), x:(x+crop_width)]

    # Rotate
    # Randomly pick an img rotate code. Do nothing if value == 0
    value = random.choice(range(4))
    if value == 1:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif value == 2:
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif value == 3:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Brighten/Darken
    # Convert to HSV
    hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Saturation and lightness change values
    # s and l can each be increased OR decreased, with 50% chance
    s = int(127 * random.choice(multipliers[:3]) * random.choice([-1, 1]))
    l = int(127 * random.choice(multipliers[:3]) * random.choice([-1, 1]))
    # Increase or decrease saturation
    hsl[:,:,1] = cv2.addWeighted(hsl[:,:,1], 1, np.zeros_like(hsl[:,:,1]), 0, s)
    # Increase or decrease brightness
    hsl[:,:,2] = cv2.addWeighted(hsl[:,:,2], 1, np.zeros_like(hsl[:,:,2]), 0, l)
    # Convert back to BGR and overwrite
    img = cv2.cvtColor(hsl, cv2.COLOR_HSV2BGR)

    images.append(img)
    print(f'Got file {i}')

# Write images
for i, image in enumerate(images):
    # Substitute the 'R' of the original filename for 'T' 
    new_filename = 'T' + files[i][1:]
    cv2.imwrite(f'./images/test1/{new_filename}', image)
print('All images modified!')