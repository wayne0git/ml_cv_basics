import cv2
import numpy as np
import mediapipe as mp

from utils.mp_util import face_detect, hand_detect
from utils.gesture_util import is_fist_closed, hand_down, hand_up, two_signal, three_signal, fetch_zoom_factor
from utils.imgproc_util import zoom_center, background_blur


# Set all status to false
video_status = False
blur_status = False
detect_face_status = False

# Set default Zoom Factor
zoom_factor = 1


def gesture_control(img):
    """
    Args:
        img: current frame

    Returns:
        frame with applied effects
    """
    # Set global variable values
    global video_status
    global zoom_factor
    global blur_status
    global detect_face_status

    # Detect hands
    result_hands = hand_detect(img)

    # Detect faces
    n_face = face_detect(img)

    # Update multi-face flag
    if n_face is None or n_face != 1:
        detect_face_status = True
    elif n_face == 1:
        detect_face_status = False

    # Update gesture control related flag
    if result_hands.multi_hand_landmarks:
        # Select first hand
        handLms = result_hands.multi_hand_landmarks[0]

        # Get hand landmark
        landmarks = handLms.landmark

        # Update video status flag (True - Black frame)
        if hand_down(landmarks):
            video_status = True
        elif hand_up(landmarks):
            video_status = False

        # Update background blur flag
        if three_signal(landmarks):
            blur_status = True
        elif two_signal(landmarks):
            blur_status = False

        # Get coordinates of index finger and thumb top
        zoom_arr = []

        for lm_id, lm in enumerate(landmarks):
            # Convert landmark coordinates to actual image coordinates
            cx, cy = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])

            # Append the coordinates
            if lm_id == 4 or lm_id == 8:
                zoom_arr.append((cx, cy))

        # Check if fingers are detected fists are closed and hand is up so video is on
        # Determine zoom factor
        if len(zoom_arr) > 1 and is_fist_closed(landmarks) and hand_up(landmarks):
            p1 = zoom_arr[0]
            p2 = zoom_arr[1]

            # Calculate the distance between two fingertips
            dist = np.sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))

            # Zoom in or out
            if 50 <= dist <= 150:
                zoom_factor = fetch_zoom_factor(dist)

    # Black frame if the hand was down or there is more than one person in frame
    if video_status is True or detect_face_status is True:
        img = np.zeros_like(img)

    # Background blur
    if blur_status:
        img = background_blur(img)

    # Zoom
    img = zoom_center(img, zoom_factor)

    return img


if __name__ == '__main__':
    # Init camera
    cap = cv2.VideoCapture(0)

    # Run gesture control
    while True:
        # Read frame
        ret, img = cap.read()

        # Gesture control
        img = gesture_control(img)

        # Show image
        cv2.imshow('Image', img)

        # Key control
        if cv2.waitKey(1) == 27:
            break
