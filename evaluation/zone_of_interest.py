import cv2
import numpy as np
from data_preprocessing.rectangular_to_circular import rect_to_circ

def generate_zone_of_interest_mask():
    """
    Generates a binary mask indicating the zone of interest.
    """
    mask = np.zeros((256, 960))
    mask[70:140, :] = 1
    mask_circ = rect_to_circ(mask)
    
    return mask_circ

def generate_zone_of_interest_image(image, mask, color=[0, 255, 0], opacity=0.5):
    """
    Generates an image with the zone of interest highlighted.
    """
    mask = mask.astype(bool)
    
    # Create an overlay with a prespecified color
    overlay = np.zeros_like(image, dtype=np.uint8)
    overlay[mask] = color
    
    # Blend overlay with the original image
    highlighted_image = cv2.addWeighted(image, 1 - opacity, overlay, opacity, 0)
    
    return highlighted_image
