import cv2
import mediapipe as mp
import numpy as np


def face_detect(image):
    """
    Args:
        image: frame captured by camera

    Returns:
        The number of faces
    """
    # Use Mediapipe face detection
    mp_fd = mp.solutions.face_detection

    # choose facedetection criteria
    with mp_fd.FaceDetection(min_detection_confidence=0.5, model_selection=0) as fd:
        # Make the image non-writeable (help improve performance)
        image.flags.writeable = False

        # Convert image from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # process the image
        results = fd.process(image)

        # If any face is detected return the number of faces
        if results.detections:
            return len(results.detections)
        else:
            return None


def hand_detect(image):
    """
    Args:
        image: frame captured by camera

    Returns:
        Hand detection results
    """
    # Use Mediapipe hand detection
    mp_hands = mp.solutions.hands

    with mp_hands.Hands() as hands:
        # Make the image non-writeable (help improve performance)
        image.flags.writeable = False

        # Convert image from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image
        results = hands.process(image)

        return results


def get_selfie_mask(image):
    """
    Args:
        image: frame captured by camera

    Returns:
        Selfie segmentation mask
    """
    # Use Mediapipe selfie segmentation
    mp_selfie_seg = mp.solutions.selfie_segmentation

    with mp_selfie_seg.SelfieSegmentation(model_selection=1) as selfie_seg:
        # Make the image non-writeable (help improve performance)
        image.flags.writeable = False

        # Convert image from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # process the image
        results = selfie_seg.process(image)

        return results.segmentation_mask
