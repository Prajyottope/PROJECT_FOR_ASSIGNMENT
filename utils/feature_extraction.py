import cv2
from skimage.feature import hog

def extract_features(image):

    image = cv2.resize(image, (64,64))

    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        transform_sqrt=True,
        block_norm="L2-Hys"
    )

    return features