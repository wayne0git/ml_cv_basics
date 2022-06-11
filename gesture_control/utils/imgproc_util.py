import cv2
import mediapipe as mp
import numpy as np

from .mp_util import get_selfie_mask


def zoom_center(image, zoom_fact):
    """
    Args:
        image: current frame
        zoom_fact: Current zoom factor

    Returns:
        zoomed image with center focus
    """
    # Get Image coordinates
    y_size = image.shape[0]
    x_size = image.shape[1]

    # define new boundaries
    x1 = int(0.5 * x_size * (1 - 1 / zoom_fact))
    x2 = int(x_size - 0.5 * x_size * (1 - 1 / zoom_fact))
    y1 = int(0.5 * y_size * (1 - 1 / zoom_fact))
    y2 = int(y_size - 0.5 * y_size * (1 - 1 / zoom_fact))

    # first crop image then scale
    img_cropped = image[y1:y2, x1:x2]
    zoomed_image = cv2.resize(img_cropped, None, fx=zoom_fact, fy=zoom_fact)

    return zoomed_image


def background_blur(image):
    """
    Args:
        image: frame captured by camera

    Returns:
        The image with a blurred background
    """
    # Get selfie segmentation mask
    seg_mask = get_selfie_mask(image)

    # Add a joint bilateral filter
    bf_img = cv2.ximgproc.jointBilateralFilter(np.uint8(seg_mask), image, 25, 5, 5)

    # Create a condition for blurring the background
    condition = np.stack((seg_mask,) * 3, axis=-1) > 0.1

    # Remove map the image on blurred background
    output_image = np.where(condition, image, bf_img)

    return output_image
