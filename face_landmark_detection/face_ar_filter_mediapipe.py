import cv2                          # opencv-contrib-python==4.5.5.64
import mediapipe as mp              # mediapipe==0.8.9.1
import numpy as np                  # numpy==1.21.5

from util import face_blend_common as fbc
from util.filter_util import FILTER_CONFIGS, load_filter
from util.landmark_util import KEYPNTS_75, get_landmarks, get_keypnts


# Parameter
FILTER_NAME = 'dog'
PLT_OPTION = 3  # 0 - Landmark (whole), 1 - Landmark (75), 2 - Face Mesh, 3 - AR filter

# Initialize MediaPipe drawing
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize video input
cap = cv2.VideoCapture(0)

# Load AR filter
iter_filter_keys = iter(FILTER_CONFIGS.keys())
filters, multi_filter_runtime = load_filter(next(iter_filter_keys))

# Some variables
isFirstFrame = True
sigma = 50

# Run face mesh
while cap.isOpened():
    # Read image
    success, image = cap.read()
    if not success:
        continue

    # Face mesh
    multi_face_landmarks = get_landmarks(image)

    # Overlay face landmark / mesh / ar filter on the image
    if multi_face_landmarks is not None:
        # Draw face landmark (Whole / 75-point)
        if PLT_OPTION == 0 or PLT_OPTION == 1:
            height, width = image.shape[:2]

            for face_landmarks in multi_face_landmarks:
                for idx, landmark in enumerate(face_landmarks.landmark):
                    if PLT_OPTION == 0 or idx in KEYPNTS_75:
                        point = (int(landmark.x * width), int(landmark.y * height))
                        cv2.circle(image, point, 1, (255, 0, 0), -1)
                        cv2.putText(image, str(idx), point, cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255), 1)
        # Draw face mesh
        elif PLT_OPTION == 2:
            mp_face_mesh = mp.solutions.face_mesh

            for face_landmarks in multi_face_landmarks:
                # Draw mesh
                mp_drawing.draw_landmarks(image=image,
                                          landmark_list=face_landmarks,
                                          connections=mp_face_mesh.FACEMESH_TESSELATION,
                                          landmark_drawing_spec=None,
                                          connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

                # Draw contour
                mp_drawing.draw_landmarks(image=image,
                                          landmark_list=face_landmarks,
                                          connections=mp_face_mesh.FACEMESH_CONTOURS,
                                          landmark_drawing_spec=None,
                                          connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

                # Draw iris
                mp_drawing.draw_landmarks(image=image,
                                          landmark_list=face_landmarks,
                                          connections=mp_face_mesh.FACEMESH_IRISES,
                                          landmark_drawing_spec=None,
                                          connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
        else:
            # Get 75-keypoint
            points2 = get_keypnts(multi_face_landmarks, image.shape[:-1])

            # Skip overlay when partially detected
            if len(points2) != 75:
                print('Face is partially detected!')
                continue

            ################ Start of Optical Flow and Stabilization Code ###############

            # Optical flow for stabilization
            img2Gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            if isFirstFrame:
                points2Prev = np.array(points2, np.float32)
                img2GrayPrev = np.copy(img2Gray)
                isFirstFrame = False

            lk_params = dict(winSize=(101, 101), maxLevel=15,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.001))
            points2Next, _, _ = cv2.calcOpticalFlowPyrLK(img2GrayPrev, img2Gray, 
                                                         points2Prev, np.array(points2, np.float32), **lk_params)

            # Weighted average of detected landmarks and tracked landmarks
            for k in range(0, len(points2)):
                d = cv2.norm(np.array(points2[k]) - points2Next[k])
                alpha = np.exp(-d * d / sigma)
                points2[k] = (1 - alpha) * np.array(points2[k]) + alpha * points2Next[k]
                points2[k] = fbc.constrainPoint(points2[k], image.shape[1], image.shape[0])
                points2[k] = (int(points2[k][0]), int(points2[k][1]))

            # Update variables for next pass
            points2Prev = np.array(points2, np.float32)
            img2GrayPrev = img2Gray

            ################ End of Optical Flow and Stabilization Code ###############

            # Overlay filter
            for idx, filter in enumerate(filters):
                # Get filter
                filter_runtime = multi_filter_runtime[idx]
                img1 = filter_runtime['img']
                img1_alpha = filter_runtime['img_a']
                points1 = filter_runtime['points']

                # Overlay
                if filter['morph']:
                    # Get filter morph info
                    dt = filter_runtime['dt']
                    hull1 = filter_runtime['hull']
                    hullIndex = filter_runtime['hullIndex']

                    # create copy of frame
                    warped_img = np.copy(image)

                    # Find convex hull
                    hull2 = []
                    for i in range(0, len(hullIndex)):
                        hull2.append(points2[hullIndex[i][0]])

                    mask1 = np.zeros((warped_img.shape[0], warped_img.shape[1]), dtype=np.float32)
                    mask1 = cv2.merge((mask1, mask1, mask1))
                    img1_alpha_mask = cv2.merge((img1_alpha, img1_alpha, img1_alpha))

                    # Warp the triangles
                    for i in range(0, len(dt)):
                        t1 = []
                        t2 = []

                        for j in range(0, 3):
                            t1.append(hull1[dt[i][j]])
                            t2.append(hull2[dt[i][j]])

                        fbc.warpTriangle(img1, warped_img, t1, t2)
                        fbc.warpTriangle(img1_alpha_mask, mask1, t1, t2)

                    # Blur the mask before blending
                    mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)
                    mask2 = (255.0, 255.0, 255.0) - mask1

                    # Perform alpha blending of the two images
                    temp1 = np.multiply(warped_img, (mask1 * (1.0 / 255)))
                    temp2 = np.multiply(image, (mask2 * (1.0 / 255)))
                    output = temp1 + temp2
                else:
                    # Apply similarity transform to input image
                    dst_points = [points2[int(list(points1.keys())[0])], points2[int(list(points1.keys())[1])]]
                    tform = fbc.similarityTransform(list(points1.values()), dst_points)

                    trans_img = cv2.warpAffine(img1, tform, (image.shape[1], image.shape[0]))
                    trans_alpha = cv2.warpAffine(img1_alpha, tform, (image.shape[1], image.shape[0]))
                    mask1 = cv2.merge((trans_alpha, trans_alpha, trans_alpha))

                    # Blur the mask before blending
                    mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)
                    mask2 = (255.0, 255.0, 255.0) - mask1

                    # Perform alpha blending of the two images
                    temp1 = np.multiply(trans_img, (mask1 * (1.0 / 255)))
                    temp2 = np.multiply(image, (mask2 * (1.0 / 255)))
                    output = temp1 + temp2

                image = np.uint8(output)
    else:
        print('Face is not detected!')

    # Show image
    cv2.imshow('MediaPipe Face Mesh', image)

    # Key control
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    elif k == ord('f'): # Change filter type
        try:
            filters, multi_filter_runtime = load_filter(next(iter_filter_keys))
        except: # Iterator end
            iter_filter_keys = iter(FILTER_CONFIGS.keys())
            filters, multi_filter_runtime = load_filter(next(iter_filter_keys))
    elif k == ord('p'): # Change overlay type
        PLT_OPTION = PLT_OPTION + 1 if PLT_OPTION < 3 else 0

cap.release()
cv2.destroyAllWindows()
