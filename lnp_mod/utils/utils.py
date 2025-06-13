import os
import glob
import cv2
import numpy as np
    
def get_supported_images_path_list(images_folder):
    """
    Get a list of paths to supported image files in the specified folder.
    
    Args:
        images_folder: Path to the folder containing images
        
    Returns:
        List of absolute paths to supported image files
    """
    image_paths = []
    # Ensure folder path ends with separator
    if not images_folder.endswith(os.sep):
        images_folder = images_folder + os.sep
        
    # Exclude '*.jpg', '*.jpeg' as this format are always used for the overview image
    supported_images = ('*.tif', '*.tiff', '*.png', '*.bmp',
                        '*.dng', '*.webp', '*.pfm', '*.mpo')
    
    for image_type in supported_images:
        pattern = os.path.join(images_folder, image_type)
        found_images = glob.glob(pattern)
        if found_images:
            image_paths.extend(found_images)
    
    # Convert all paths to absolute paths
    image_paths = [os.path.abspath(path) for path in image_paths]

    return sorted(image_paths)

def disable_jpg_images(folder):
    for image in glob.glob(folder + '*.jpg'):
        os.rename(image, image + '.disable')

def restore_jpg_images(folder):
    for image in glob.glob(folder + '*.disable'):
        os.rename(image, image[:-8])

def create_output_folder(target_dir):
    counter = 0
    output_folder_name = 'result'
    output_folder = os.path.join(target_dir, output_folder_name, '')

    while os.path.exists(output_folder):
        counter += 1
        output_folder = os.path.join(target_dir, output_folder_name + str(counter), '')

    os.makedirs(output_folder)
    return output_folder


def compute_dice_coefficient(gt_mask, pred_mask):
    # Convert masks to boolean type
    gt_mask = gt_mask.astype(np.bool_)
    pred_mask = pred_mask.astype(np.bool_)

    # Calculate the intersection and the sum of elements in both masks
    intersection = np.count_nonzero(gt_mask & pred_mask)
    sum_masks = np.count_nonzero(gt_mask) + np.count_nonzero(pred_mask)

    # Compute the Dice coefficient
    dice_coefficient = (2 * intersection) / sum_masks if sum_masks != 0 else 1.0
    
    return dice_coefficient


def mask_to_single_polygon(mask, simplify=False, epsilon=1.0):
    """Convert a binary mask to a polygon (largest contour).
    
    Args:
        mask (np.ndarray): Binary mask
        simplify (bool): Whether to simplify the polygon
        epsilon (float): Parameter for polygon simplification (higher = more simplification)
        
    Returns:
        list: List containing a single polygon in COCO format or empty list if no valid contours
    """
    # Validate input mask
    if mask is None or mask.size == 0:
        print("Warning: Empty mask provided to mask_to_single_polygon")
        return []
    
    # Ensure mask is binary and uint8 type (required by findContours)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    
    # Make sure mask contains only 0s and 1s/255s
    if np.max(mask) > 1:
        mask = (mask > 0).astype(np.uint8) * 255
    else:
        mask = (mask > 0).astype(np.uint8) * 255
    
    # Check if mask is entirely empty (all zeros)
    if np.sum(mask) == 0:
        print("Warning: Mask is completely empty (all zeros)")
        return []
    
    try:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("Warning: No contours found in non-empty mask")
            return []
        
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Check if contour has enough points to form a polygon
        if len(largest_contour) < 3:
            print(f"Warning: Largest contour has only {len(largest_contour)} points, not enough for a polygon")
            return []
        
        # Simplify polygon if requested
        if simplify and len(largest_contour) > 10:  # Only simplify if enough points
            largest_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Convert contour to COCO format (flattened list of x,y coordinates)
        polygon = largest_contour.flatten().tolist()
        
        # Ensure we have at least 6 points (3 points x,y coordinates) for a valid polygon
        if len(polygon) < 6:
            print(f"Warning: Polygon has only {len(polygon)//2} points, need at least 3 for valid polygon")
            return []
        
        # Ensure we have an even number of coordinates (x,y pairs)
        if len(polygon) % 2 != 0:
            print(f"Warning: Odd number of values in polygon: {len(polygon)}")
            polygon = polygon[:-1]  # Remove last element to make it even
        
        # Return as a list of polygons, containing just the largest one
        return [polygon]
    
    except Exception as e:
        print(f"Error in mask_to_single_polygon: {str(e)}")
        return []

def imread(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def imwrite(path, image):
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))