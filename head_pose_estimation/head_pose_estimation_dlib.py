# Ref 
# 1. https://github.com/lincolnhard/head-pose-estimation/blob/master/video_test_shape.py
# 2. https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
# 3. https://blog.gtwang.org/programming/python-opencv-dlib-face-detection-implementation-tutorial/
# 4. https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
# 5. https://github.com/jerryhouuu/Face-Yaw-Roll-Pitch-from-Pose-Estimation-using-OpenCV

import cv2
import dlib
import math
import numpy as np

from imutils import face_utils

# Trained model downloaded from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
face_landmark_path = './shape_predictor_68_face_landmarks.dat'

# Landmark index used for pose estimation (2D projection)
pts_idx_pose = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]

# Landmark 3D coordinates
object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])

# Intrinsic parameters of the camera (Approximated)
# cam_matrix = [fx, 0, cx ; 0, fy, cy ; 0, 0, 1] 
#              (fx = fy = IM_W / 2, cx = IM_W / 2, cy = IM_H / 2)
# dist_coeffs -- Distortion coefficients (Assuming no lens distortion)
cam_matrix = np.array([[640 ,0  , 320],\
                       [0   ,640, 240],\
                       [0   ,0  , 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))

# 3D coordinate of cube for pose visualization
reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]

def get_head_pose(shape):
    # Extract points for pose estimation
    image_pts = np.float32([shape[idx] for idx in pts_idx_pose])
    
    # Solve PnP problem for pose estimation (rotaion / translation)
    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    # Calculate euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)   # (3, 1) -> (3, 3)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat) # (3, 1)

    # Project simulated cube for pose visualization
    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix, dist_coeffs)
    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    return reprojectdst, euler_angle

def main():
    # Read from camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to connect to camera.")
        return
    
    # Class object for face / landmark detection
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_landmark_path)

    # Process each frame
    cv2.namedWindow("Pose Estimation", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Face detection (Based on 5 HOG filters for different face orientation)
            # Input -- (Image data, # of upsample, Score threshold)
            # Output -- (dlib rects, Score of each rect, Detector index of each rect)
            face_rects, scores, idx = detector.run(frame, 0, -0.5)

            # Find face landmarks within each face ROI
            for (i, rect) in enumerate(face_rects):
                # Determine facial landmarks 
                # Convert the facial landmark (x, y)-coordinates to a NumPy array
                shape = predictor(frame, rect)
                shape = face_utils.shape_to_np(shape)

                # Compute head pose
                if i == 0:
                    reprojectdst, euler_angle = get_head_pose(shape)

                # Draw face detection result
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Face #%d (%.2f / %d)" % (i, scores[i], idx[i]), (x - 10, y - 10),\
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
             
                # Draw face landmark detection result (Only draw landmarks for pose estimation)
                for pts_idx in pts_idx_pose:
                    cv2.circle(frame, (shape[pts_idx][0], shape[pts_idx][1]), 3, (0, 0, 255), -1)

                # Draw head pose estimation result
                if i == 0:
                    pitch, yaw, roll = euler_angle
                    for start, end in line_pairs:
                        cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))
    
                    cv2.putText(frame, "Pitch: %7.2f" % pitch, (20, 20),\
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)
                    cv2.putText(frame, "Roll: %7.2f" % roll, (20, 50),\
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)
                    cv2.putText(frame, "Yaw: %7.2f" % yaw, (20, 80),\
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)

            cv2.imshow("Pose Estimation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()
