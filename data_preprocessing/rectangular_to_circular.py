import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

def rect_to_circ_rgb(img_rgb_rect):
    """Convert a rectangular image to a circular image (3 channels)."""
    img_rgb_circ = np.empty((256, 256, 3))
    for i in (0, 1, 2):
        img_rgb_circ[:, :, i] = rect_to_circ(img_rgb_rect[:, :, i])

    return img_rgb_circ

def rect_to_circ(img_rect):
    """Convert a rectangular image to a circular image (1 channel)."""
    # Initialize the parameters
    r_min, r_max = 0, 1
    mr, nr = 256, 256

    # Initialize the output image
    img_circ = np.zeros((mr, nr))

    # Calculate center of the image
    om = (mr + 1) / 2
    on = (nr + 1) / 2

    # Scale factors for x and y coordinates
    sx = (mr - 1) / 2
    sy = (nr - 1) / 2

    # Get dims of the rectangular image
    m, n = img_rect.shape

    # Calculate the increments in the radius and angle
    del_r = (r_max - r_min) / (m - 1)
    del_t = 2 * np.pi / n

    # Iterate over each pixel in the output (circular) image
    for xi in range(mr):
        for yi in range(nr):
            x = (xi + 1 - om) / sx
            y = (yi + 1 - on) / sy
            r = np.sqrt(x ** 2 + y ** 2)
            if r_min <= r <= r_max:
                t = np.arctan2(y, x)
                if t < 0:
                    t += 2 * np.pi
                img_circ[xi, yi] = interpolate(img_rect, r, t, r_min, r_max, m, n, del_r, del_t)

    return img_circ


def interpolate(im_p, r, t, r_min, r_max, m, n, del_r, del_t):
    # Calculate the fractional indices in the rectangular image
    ri = 1 + (r - r_min) / del_r
    ti = 1 + t / del_t
    rf, rc = int(np.floor(ri)), int(np.ceil(ri))
    tf, tc = int(np.floor(ti)), int(np.ceil(ti))

    # Handle boundary conditions
    if tc > n:
        tc = tf

    # Compute simple, linear or bilinear interpolation
    if rf == rc and tc == tf:
        return im_p[rf-1, tc-1]
    elif rf == rc:
        return im_p[rf-1, tf-1] + (ti - tf) * (im_p[rf-1, tc-1] - im_p[rf-1, tf-1])
    elif tf == tc:
        return im_p[rf-1, tf-1] + (ri - rf) * (im_p[rc-1, tf-1] - im_p[rf-1, tf-1])
    else:
        a = np.array([
            [rf, tf, rf * tf, 1],
            [rf, tc, rf * tc, 1],
            [rc, tf, rc * tf, 1],
            [rc, tc, rc * tc, 1]
        ])
        z = np.array([
            im_p[rf-1, tf-1],
            im_p[rf-1, tc-1],
            im_p[rc-1, tf-1],
            im_p[rc-1, tc-1]
        ])
        coeffs = np.linalg.lstsq(a, z, rcond=None)[0]
        v = np.array([ri, ti, ri * ti, 1]) @ coeffs
        return v