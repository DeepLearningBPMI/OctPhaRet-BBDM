import numpy as np

def normalize(img, min_val=0, max_val=2):
    """Clip values outside of min_val and max_val, normalize to [0, 1] and scale to [0, 255]."""
    img = np.clip(img, max_val)
    img = (img - min_val) / (max_val - min_val)
    img = img * 255
    return img