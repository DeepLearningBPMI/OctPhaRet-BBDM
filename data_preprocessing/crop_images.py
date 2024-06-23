import numpy as np

def crop_img_circ(img_circ, mask_circ):
    """Use binary mask to crop a circular image. To get mask with a ring, use generate_zone_of_interest_mask() from evaluation/zone_of_interest.py."""
    # mask_circ = np.load('./crop_mask_circ.npy')
    if len(img_circ.shape) == 2:
        return img_circ * mask_circ
    else:
        return img_circ * np.repeat(mask_circ[:, :, np.newaxis], 3, axis=2)
