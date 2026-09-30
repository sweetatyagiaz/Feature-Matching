import cv2


class SIFTDetector:
    def __init__(self, nfeatures=5000):
        self.detector = cv2.SIFT_create(
            nfeatures=nfeatures
        )

    def extract(self, image):
        keypoints, descriptors = self.detector.detectAndCompute(
            image,
            None
        )

        return keypoints, descriptors

