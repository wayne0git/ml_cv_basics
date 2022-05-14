import mediapipe as mp
import cv2
import numpy as np


# Landmark index for AR filter
KEYPNTS_75 = [127, 93, 58, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 288, 323, 356, 70, 63, 105, 66, 55,
              285, 296, 334, 293, 300, 168, 6, 195, 4, 64, 60, 94, 290, 439, 33, 160, 158, 173, 153, 144, 398, 385, 387,
              466, 373, 380, 61, 40, 39, 0, 269, 270, 291, 321, 405, 17, 181, 91, 78, 81, 13, 311, 306, 402, 14, 178,
              162, 54, 67, 10, 297, 284, 389]


# Detect facial landmarks based on Mediapipe (Assume BGR input image)
def get_landmarks(img):
    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, 
                               min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        # Landmark detection
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        return results.multi_face_landmarks


def get_keypnts(multi_face_landmarks, img_shape, face_ind=0):
    # Select one face
    face_landmarks = multi_face_landmarks[face_ind]

    # Convert normalized points to image coordinates
    values = np.array(face_landmarks.landmark)
    face_keypnts = np.zeros((len(values), 2))

    for idx, value in enumerate(values):
        face_keypnts[idx][0] = value.x
        face_keypnts[idx][1] = value.y

    face_keypnts = face_keypnts * (img_shape[1], img_shape[0])
    face_keypnts = face_keypnts.astype('int')

    # Get 75-keypoints
    relevant_keypnts = [face_keypnts[i] for i in KEYPNTS_75]

    return relevant_keypnts

