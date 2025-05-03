import cv2
import numpy as np

# Define a function to compute Intersection Over Union
def compute_iou(mask1, mask2):
    # Calculate the overlap and total area masks
    overlap = np.logical_and(mask1, mask2)
    total_area = np.logical_or(mask1, mask2)
    
    # Compute IoU as the ratio of overlap to total area
    iou = np.sum(overlap) / np.sum(total_area)
    return iou

def main_function():
    # Load the ground truth image (binary image with cracks)
    gt_image_path = '/Users/sachinsajith/Downloads/crack4_groundtruth.png'
    ground_truth = cv2.imread(gt_image_path, 0) / 255.0

    # Load the detected crack image (binary image)
    detected_image_path = '/Users/sachinsajith/Downloads/morph_processed_crack4.png'
    detected_crack = cv2.imread(detected_image_path, 0) / 255.0

    # Compute IoU between predicted and ground truth masks
    iou_score = compute_iou(detected_crack, ground_truth)

    # Print the IoU score with 4 decimal places
    print(f"IoU (Intersection over Union) Score: {iou_score:.4f}")

if __name__ == "__main__":
    main_function()
