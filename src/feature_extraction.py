import cv2


def extract_hog_features(image):
    """
    Extracts HOG features from the input image.

    Args:
    - image (numpy array): a numpy array of shape (H, W) representing the grayscale image.

    Returns:
    - features (numpy array): a numpy array of shape (N,) containing the HOG features.
    """
    # Define the HOG parameters
    win_size = (64, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    num_bins = 9

    # Create the HOG descriptor
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)

    # Compute the HOG features
    features = hog.compute(image)

    return features


def extract_color_hist(image):
    """
    Extracts color histogram features from the input image.

    Args:
    - image (numpy array): a numpy array of shape (H, W, C) representing the color image.

    Returns:
    - features (numpy array): a numpy array of shape (N,) containing the color histogram features.
    """
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Compute the color histogram features
    hist = cv2.calcHist(
        images=[hsv_image],
        channels=[0, 1, 2],
        mask=None,
        histSize=[8, 8, 8],
        ranges=[0, 256, 0, 256, 0, 256]
    )
    hist = cv2.normalize(hist, hist).flatten()

    return hist
