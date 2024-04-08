import cv2
import numpy as np
from skimage.feature import hog


def extract_hog_features(image, return_img=False):
    """
    This function computes HOG features for a 64x64 RGB image.

    Args:
      image: A numpy array of shape (64, 64, 3) representing the RGB image.

    Returns:
      A numpy array representing the HOG features of the image.
    """
    num_orientations = 8
    pixels_per_cell = (8, 8)
    cells_per_block = (2, 2)

    # Check if the image has the expected dimensions
    if image.shape[0] != 3:
        image = image.transpose(1, 2, 0)
        if image.shape[0] != 3:
            raise ValueError("Image must start with channel dimension.")

    # Convert the image to grayscale (HOG is typically computed on grayscale images)
    gray_image = image.mean(axis=0)

    # Compute HOG features using scikit-image
    hog_features, hog_image = hog(
        gray_image,
        orientations=num_orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=True
    )
    if return_img:
        return hog_features, hog_image

    return hog_features


def extract_color_hist(image):
    """
    This function computes the color histogram for each channel (RGB) in an image.

    Args:
      image: A numpy array representing the RGB image.

    Returns:
      A numpy array of shape (3, 256) containing the histograms for R, G, and B channels.
    """
    num_bins = 100
    range = (0, 256)

    # Check if the image has the expected dimensions
    if image.shape[-1] != 3:
        image = image.transpose(1, 2, 0)
        if image.shape[-1] != 3:
            raise ValueError("Image must start with channel dimension.")

    # Split the image into separate channels
    channels = np.split(image, 3, axis=2)

    # Compute histogram for each channel (256 bins)
    histograms = [np.histogram(channel.flatten(), bins=num_bins, range=range)[0] for channel in channels]

    # Stack the histograms for each channel
    return np.stack(histograms, axis=0)
