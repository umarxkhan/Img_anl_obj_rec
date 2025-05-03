import cv2
import numpy as np
from scipy.ndimage import maximum_filter
import math
from skimage.morphology import skeletonize, label
from scipy.spatial.distance import euclidean

# Function for non-maximum suppression of magnitude and angle data
def find_peak_values(magnitude, angle):
    # Quantize angles to four directions (0, 45, 90, and 135 degrees)
    quantized_angle = np.round(angle / (np.pi / 4)) % 4
    # Define 3x3 windows for each direction
    directional_windows = [
        np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]),
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]]),
        np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    ]
    # Apply non-maximum suppression for each direction
    for i in range(4):
        magnitude[quantized_angle == i] = apply_suppression(magnitude, directional_windows[i])[quantized_angle == i]
    return magnitude

# Function to perform non-maximum suppression
def apply_suppression(data, window):
    data_max = maximum_filter(data, footprint=window, mode='constant')
    data_max[data != data_max] = 0
    return data_max

# Function to compute the length and normalized length of detected cracks
def compute_normalized_crack_length(crack_image):
    # Get image dimensions
    image_height, image_width = crack_image.shape
    
    # Compute the total number of pixels in the image
    total_pixels = image_height * image_width
    
    # Get coordinates of all crack pixels
    crack_coordinates = np.column_stack(np.where(crack_image == 1))
    
    # Initialize the variable to hold the total crack length
    total_length = 0.0
    
    # For each crack pixel, find its adjacent pixels and compute the distance
    for coord in crack_coordinates:
        x, y = coord
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                adjacent_x, adjacent_y = x + dx, y + dy
                if 0 <= adjacent_x < image_height and 0 <= adjacent_y < image_width:
                    if crack_image[adjacent_x, adjacent_y] == 1:
                        total_length += euclidean(coord, [adjacent_x, adjacent_y])
                        
    # To avoid counting the distance twice, divide by 2
    total_length /= 2.0
    
    # Normalize the crack length by the total number of pixels
    normalized_length = "{:e}".format(total_length / total_pixels)
    
    return normalized_length

# Function to process and analyze cracks in an image
def analyze_cracks_in_image(image_path):
    # Load the image
    grayscale_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) / 255.0

    # Display the input image
    cv2.imshow('Input Image', grayscale_image)
    cv2.waitKey()

    # Gaussian blur parameters
    sigma = 21
    kernel_size = 2 * math.ceil(2 * sigma) + 1

    # Apply Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(grayscale_image, (kernel_size, kernel_size), sigma)
    grayscale_image = cv2.subtract(grayscale_image, blurred_image)

    # Compute the gradients using Sobel operators
    gradient_x = cv2.Sobel(grayscale_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(grayscale_image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute gradient magnitude and angle
    magnitude = np.hypot(gradient_x, gradient_y)
    angle = np.arctan2(gradient_y, gradient_x)

    # Thresholding to remove weak edges
    threshold = 4 * 1.3 * np.mean(magnitude)
    magnitude[magnitude < threshold] = 0

    # Apply non-maximum suppression to the magnitude
    magnitude = find_peak_values(magnitude, angle)

    # Threshold the magnitude to create a binary image
    magnitude[magnitude > 0] = 255
    magnitude = magnitude.astype(np.uint8)

    # Display the thresholded image
    cv2.imshow('Thresholded Image', magnitude)
    cv2.waitKey()

    # Connected components labeling
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(magnitude, connectivity=8)

    # Remove small connected components (noise)
    min_area = 50
    for i in range(1, len(stats)):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            labels[labels == i] = 0

    # Dynamic Morphological Operations based on average crack area
    average_area = np.mean(stats[:, cv2.CC_STAT_AREA])
    kernel_size = 3 if average_area < 50 else 5
    morphological_kernel = np.ones((kernel_size, kernel_size), np.uint8)

    morphed_image = cv2.morphologyEx(magnitude, cv2.MORPH_CLOSE, morphological_kernel)

    # Thinning using skeletonize for crack analysis
    filtered_image = skeletonize(morphed_image)

    # Dilate the skeleton to make it softer
    dilate_kernel = np.ones((3, 3), np.uint8)
    filtered_image = cv2.dilate(filtered_image.astype(np.uint8), dilate_kernel, iterations=1)

    # Post-processing to remove noise by defining a minimum crack size
    min_crack_size = 45

    # Remove small connected components (noise) in the thinned image
    labeled_cracks, num_features = label(filtered_image, connectivity=2, return_num=True)
    for label_id in range(1, num_features + 1):
        if np.sum(labeled_cracks == label_id) < min_crack_size:
            filtered_image[labeled_cracks == label_id] = 0

    # Display the image after morphological processes
    cv2.imshow('Processed Image', filtered_image.astype(np.uint8) * 255)
    cv2.waitKey()

    return filtered_image, stats

# Function to draw rectangles around identified cracks
def draw_crack_rectangles(image_path, is_crack, stats):
    output_image = cv2.imread(image_path)
    for i in range(1, len(stats)):
        if is_crack[i - 1]:
            x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
            width, height = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            cv2.rectangle(output_image, (x, y), (x + width, y + height), (0, 255, 0), 2)

    # Display the identified cracks
    cv2.imshow('Identified Cracks', output_image)
    cv2.waitKey()

# Main function to analyze cracks in an image
def main():
    image_path = '/Users/sachinsajith/Documents/Bauhaus Universität/SoSe23/Image Analysis/Project/crack_detection/input_images/crack1.png'
    filtered_image, stats = analyze_cracks_in_image(image_path)

    # Extract features of connected components
    features = []
    for i in range(1, len(stats)):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        aspect_ratio = float(width) / height
        circularity = 4 * np.pi * (area / (width * height))

        features.append([area, aspect_ratio, circularity])

    features = np.array(features)

    # Compute mean and standard deviation of features
    mean_area, std_area = np.mean(features[:, 0]), np.std(features[:, 0])
    mean_aspect_ratio, std_aspect_ratio = np.mean(features[:, 1]), np.std(features[:, 1])
    mean_circularity, std_circularity = np.mean(features[:, 2]), np.std(features[:, 2])

    # Define thresholds and weights
    thresholds = {
        'area': mean_area - 0.5 * std_area,
        'aspect_ratio': mean_aspect_ratio + 0.5 * std_aspect_ratio,
        'circularity': mean_circularity - 0.5 * std_circularity
    }

    threshold_weight = {
        'area': 0.5,
        'aspect_ratio': 0.3,
        'circularity': 0.2
    }

    # Identify cracks based on weighted feature thresholds
    is_crack = []
    for feature in features:
        weighted_sum = 0
        area, aspect_ratio, circularity = feature
        weighted_sum += threshold_weight['area'] * (area > thresholds['area'])
        weighted_sum += threshold_weight['aspect_ratio'] * (aspect_ratio < thresholds['aspect_ratio'])
        weighted_sum += threshold_weight['circularity'] * (circularity > thresholds['circularity'])

        if weighted_sum >= 0.7:  # The sum exceeds the weighted threshold
            is_crack.append(True)
        else:
            is_crack.append(False)

    # Draw rectangles around identified cracks
    draw_crack_rectangles(image_path, is_crack, stats)

    # Step 1: Start with the thinned image
    filtered_image = np.copy(filtered_image)

    # Step 2: Blacken the non-crack regions
    for i in range(1, len(stats)):
        if not is_crack[i - 1]:  # If this connected component is NOT identified as a crack
            x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
            width, height = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            filtered_image[y:y + height, x:x + width] = 0  # Set these pixels to black

    # Step 3: Display the filtered image
    cv2.imshow('Filtered Image', (filtered_image * 255).astype(np.uint8))
    cv2.waitKey()

    # Compute and report the length of the detected cracks
    crack_length = compute_normalized_crack_length(filtered_image)  # Use filtered_image here
    print(f"Length of the detected cracks normalized for image resolution: {crack_length} normalized units.")

if __name__ == "__main__":
    main()
