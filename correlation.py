import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import torch
### dataset correlation
path_to_UIEB_dataset = '/home/reference-890'
image_files = os.listdir(path_to_UIEB_dataset)

# This will store the correlation matrices of all images
correlation_matrices = []

# Process each image in the dataset
for image_file in image_files:
    # Construct the full image path
    image_path = os.path.join(path_to_UIEB_dataset, image_file)

    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Flatten each channel
    r = image[:, :, 0].flatten()
    g = image[:, :, 1].flatten()
    b = image[:, :, 2].flatten()

    # Compute the correlation matrix for the current image
    corr_matrix = np.corrcoef([r, g, b])

    # Append the correlation matrix to the list
    correlation_matrices.append(corr_matrix)

# Compute the average correlation matrix across all images
average_corr_matrix = np.mean(correlation_matrices, axis=0)

# Plot the average correlation matrix
sns.heatmap(average_corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', xticklabels=['R', 'G', 'B'],
            yticklabels=['R', 'G', 'B'])
plt.title('Average Correlation patterns among RGB channels in UIEB Dataset')
plt.show()
