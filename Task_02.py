import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque

def region_growing(image, seeds, threshold):
    """Implement region growing algorithm"""
    segmented = np.zeros_like(image)
    h, w = image.shape
    queue = deque()
    
    for seed in seeds:
        queue.append(seed)
        segmented[seed] = 1
    
    neighbors = [(-1,-1), (-1,0), (-1,1),
                 (0,-1),          (0,1),
                 (1,-1),  (1,0),  (1,1)]
    
    while queue:
        x, y = queue.popleft()
        
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < h and 0 <= ny < w:
                if segmented[nx, ny] == 0 and abs(int(image[nx, ny]) - int(image[x, y])) <= threshold:
                    segmented[nx, ny] = 1
                    queue.append((nx, ny))
    
    return segmented

def create_test_image():
    """Create a test image if no image is provided"""
    image = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(image, (100, 100), 50, 100, -1)
    cv2.circle(image, (150, 150), 40, 200, -1)
    return image

def display_results(original, segmented):
    """Display the segmentation results"""
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(original, cmap='gray'), plt.title('Original Image')
    plt.subplot(132), plt.imshow(segmented, cmap='gray'), plt.title('Segmented Region')
    plt.subplot(133), plt.imshow(original, cmap='gray')
    plt.imshow(segmented, alpha=0.3), plt.title('Overlay')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image = cv2.imread('sample.jpg', cv2.IMREAD_GRAYSCALE) if cv2.imread('sample.jpg') is not None else create_test_image()
    seeds = [(100, 100), (150, 150)] 
    threshold = 20
    segmented_region = region_growing(image, seeds, threshold)
    display_results(image, segmented_region)