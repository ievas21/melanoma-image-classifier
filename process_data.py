## melanoma-classifier/process_data.py

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Standard image size for image resizing (50 by 50 pixels)
IMG_SIZE = 50

# One-hot encoding for the benign and malignant classes
BENIGN = np.array([1, 0])
MALIGNANT = np.array([0, 1])

# Paths to the benign and malignant training datasets for the model
benign_train_path = 'melanoma_cancer_dataset/train/benign'
malignant_train_path = 'melanoma_cancer_dataset/train/malignant'

# Paths to the benign and malignant testing datasets for the model
benign_test_path = 'melanoma_cancer_dataset/test/benign'
malignant_test_path = 'melanoma_cancer_dataset/test/malignant'

# Lists to hold the training and testing data
benign_train_data = []
malignant_train_data = []

benign_test_data = []
malignant_test_data = []

def load_images(path):
    '''
    Loads images from the specified path and appends the images and the one-hot encoding
    to the corresponding list.

    Args:
        path (str): The path to the directory containing the images.
    '''
    for filename in os.listdir(path):
        try:
            # Concatenate the filename with the path
            img_path = path + '/' + filename
            
            # Read the image file located at the path (GRAYSCALE for simplicity)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Failed to load {img_path}")
                continue

            # Resize the image to the standard size (50 by 50 pixels)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img_array = np.array(img)

            # Append the image array and its label to the data list
            # The label is represented as a one-hot encoded vector
            if 'benign' in path:
                if 'train' in path:
                    benign_train_data.append([img_array, BENIGN])
                else:
                    benign_test_data.append([img_array, BENIGN])

            elif 'malignant' in path:
                if 'train' in path:
                    malignant_train_data.append([img_array, MALIGNANT])
                else:
                    malignant_test_data.append([img_array, MALIGNANT])

            # print(f"Processed file: {filename} from {path}")

        except Exception as e:
            print(f"Error processing file {filename}: {e}")

# Create the benign and malignant training/testing datasets
datasets = [benign_train_path, malignant_train_path, benign_test_path, malignant_test_path]
for dataset in datasets:
    load_images(dataset)

# Ensure the training datasets are balanced by truncating the larger dataset
if len(benign_train_data) > len(malignant_train_data):
    benign_train_data = benign_train_data[:len(malignant_train_data)]
elif len(malignant_train_data) > len(benign_train_data):
    malignant_train_data = malignant_train_data[:len(benign_train_data)]

# Combine the benign and malignant datasets into a single list
training_data = benign_train_data + malignant_train_data
testing_data = benign_test_data + malignant_test_data

# Shuffle the data to ensure randomness
np.random.shuffle(training_data)
np.random.shuffle(testing_data)

# Create the output directory for processed data
out_dir = Path("data/processed")
out_dir.mkdir(parents=True, exist_ok=True)

# Define the paths for saving the training and testing data
training_path = out_dir / "melanoma_training_data.npy"
testing_path  = out_dir / "melanoma_testing_data.npy"

# Save the training and testing data as a Python object in .npy format
np.save(training_path, np.array(training_data, dtype=object), allow_pickle=True)
np.save(testing_path, np.array(testing_data, dtype=object), allow_pickle=True)
