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

# Iterate over all files in the benign training dataset directory
for filename in os.listdir(benign_train_path):
    
    try:
        # Concatenate the filename with the benign training path
        path = benign_train_path + '/' + filename
        
        # Read the image file located at the path (GRAYSCALE for simplicity)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Failed to load {path}")
            continue

        # Resize the image to the standard size (50 by 50 pixels)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_array = np.array(img)

        # Append the image array and its label (benign) to the training data list
        # The label is represented as a one-hot encoded vector
        benign_train_data.append([img_array, BENIGN])

        # plt.imshow(img)
        # plt.show()
        # break
        # print(f"Processed benign file: {filename}")

    except Exception as e:
        print(f"Error processing file {filename}: {e}")

# Iterate over all files in the malignant training dataset directory
for filename in os.listdir(malignant_train_path):
    
    try:
        # Concatenate the filename with the malignant training path
        path = malignant_train_path + '/' + filename
        
        # Read the image file located at the path (GRAYSCALE for simplicity)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Failed to load {path}")
            continue

        # Resize the image to the standard size (50 by 50 pixels)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_array = np.array(img)

        # Append the image array and its label (malignant) to the training data list
        # The label is represented as a one-hot encoded vector
        malignant_train_data.append([img_array, MALIGNANT])

    except Exception as e:
        print(f"Error processing file {filename}: {e}")

# Iterate over all files in the benign testing dataset directory
for filename in os.listdir(benign_test_path):
    
    try:
        # Concatenate the filename with the benign testing path
        path = benign_test_path + '/' + filename
        
        # Read the image file located at the path (GRAYSCALE for simplicity)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Failed to load {path}")
            continue

        # Resize the image to the standard size (50 by 50 pixels)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_array = np.array(img)

        # Append the image array and its label (benign) to the testing data list
        # The label is represented as a one-hot encoded vector
        benign_test_data.append([img_array, BENIGN])

        # plt.imshow(img)
        # plt.show()
        # break

    except Exception as e:
        print(f"Error processing file {filename}: {e}")

# Iterate over all files in the malignant testing dataset directory
for filename in os.listdir(malignant_test_path):
    
    try:
        # Concatenate the filename with the malignant testing path
        path = malignant_test_path + '/' + filename
        
        # Read the image file located at the path (GRAYSCALE for simplicity)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Failed to load {path}")
            continue

        # Resize the image to the standard size (50 by 50 pixels)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_array = np.array(img)

        # Append the image array and its label (malignant) to the testing data list
        # The label is represented as a one-hot encoded vector
        malignant_test_data.append([img_array, MALIGNANT])

        # plt.imshow(img)
        # plt.show()
        # break

    except Exception as e:
        print(f"Error processing file {filename}: {e}")

# Ensure the training datasets are balanced by truncating the larger dataset
if len(benign_train_data) > len(malignant_train_data):
    benign_train_data = benign_train_data[:len(malignant_train_data)]
elif len(malignant_train_data) > len(benign_train_data):
    malignant_train_data = malignant_train_data[:len(benign_train_data)]

# print(len(benign_train_data), "benign training images processed")
# print(len(malignant_train_data), "malignant training images processed")
# print(len(benign_test_data), "benign testing images processed")
# print(len(malignant_test_data), "malignant testing images processed")

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
