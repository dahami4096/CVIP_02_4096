import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.util import random_noise

def create_test_image():
    """Create a test image with 2 objects and background"""
    image = np.zeros((256, 256), dtype=np.uint8)
    image[50:150, 50:150] = 100  # Object 1
    image[150:200, 150:200] = 200  # Object 2
    return image

def apply_otsu(image):
    """Apply Otsu's thresholding to an image"""
    
    noisy_image = random_noise(image, mode='gaussian', var=0.01)
    noisy_image = np.array(255*noisy_image, dtype=np.uint8)

    ret, otsu_thresh = cv2.threshold(noisy_image, 0, 255, 
                                    cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return noisy_image, otsu_thresh, ret

def display_results(original, noisy, thresholded, threshold_value):
    """Display the processing results"""
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(original, cmap='gray'), plt.title('Original Image')
    plt.subplot(132), plt.imshow(noisy, cmap='gray'), plt.title('Noisy Image')
    plt.subplot(133), plt.imshow(thresholded, cmap='gray'), 
    plt.title(f'Otsu Threshold: {threshold_value}')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    original_image = create_test_image()
    noisy_img, otsu_img, thresh_val = apply_otsu(original_image)
    display_results(original_image, noisy_img, otsu_img, thresh_val)