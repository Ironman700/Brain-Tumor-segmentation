import cv2
import numpy as np
from skimage import filters, segmentation
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def multiscale_otsu(image):
    thresholds = filters.threshold_multiotsu(image, classes=3)
    regions = np.digitize(image, bins=thresholds)
    return regions

def edge_aware_smoothing(image):
    smoothened_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    return smoothened_image

def knn_segmentation(original_image, smoothened_image):
    original_flat = original_image.flatten().reshape(-1, 1)
    smoothened_flat = smoothened_image.flatten().reshape(-1, 1)
    
    X = np.concatenate([original_flat, smoothened_flat], axis=1)
    y = multiscale_otsu(smoothened_image).flatten()
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)
    
    segmented = knn.predict(X).reshape(original_image.shape)
    return segmented

def region_growing(image, seed):
    mask = np.zeros_like(image)
    stack = [seed]
    
    while stack:
        x, y = stack.pop()
        if mask[x, y] == 0:
            mask[x, y] = 1
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if 0 <= x + dx < image.shape[0] and 0 <= y + dy < image.shape[1]:
                        if abs(int(image[x, y]) - int(image[x + dx, y + dy])) < 20: # Threshold for growing
                            stack.append((x + dx, y + dy))
    return mask

# Load MRI Image
image = cv2.imread('mri_image.png', cv2.IMREAD_GRAYSCALE)

# Edge-Aware Smoothing
smoothened_image = edge_aware_smoothing(image)

# Multiscale Otsu Segmentation
segmented_image = multiscale_otsu(smoothened_image)

# KNN Segmentation
knn_segmented_image = knn_segmentation(image, smoothened_image)

# Region Growing from a Seed Point
seed_point = (100, 100)  # Example seed point, adjust accordingly
tumor_region = region_growing(knn_segmented_image, seed_point)

# Visualization
plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.subplot(1, 4, 2)
plt.title('Smoothened Image')
plt.imshow(smoothened_image, cmap='gray')
plt.subplot(1, 4, 3)
plt.title('KNN Segmented Image')
plt.imshow(knn_segmented_image, cmap='gray')
plt.subplot(1, 4, 4)
plt.title('Tumor Region')
plt.imshow(tumor_region, cmap='gray')
plt.show()
