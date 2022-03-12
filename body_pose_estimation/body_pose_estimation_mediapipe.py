import cv2
import time
import math as m
import mediapipe as mp


# Constant
FONT = cv2.FONT_HERSHEY_SIMPLEX

RED = (50, 50, 255)
GREEN = (127, 255, 0)
LIGHT_GREEN = (127, 233, 100)
YELLOW = (0, 255, 255)
PINK = (255, 0, 255)


def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2-x1)**2+(y2-y1)**2)
    return dist


def findAngle(x1, y1, x2, y2):
    theta = m.acos( (y2 -y1)*(-y1) / (findDistance(x1, y1, x2, y2) * y1) )
    degree = int(180/m.pi)*theta
    return degree


if __name__ == "__main__":
    # Initialize mediapipe pose class.
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Initialize video
    cap = cv2.VideoCapture(0)

    # Initialize image window
    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)

    # Run Mediapipe
    while True:
        # Capture frames
        success, image = cap.read()
        if not success:
            print("Null.Frames")
            break

        # Get height and width of the frame.
        h, w = image.shape[:2]

        # Convert the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image.
        keypoints = pose.process(image)

        # Convert the image back to BGR.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Use lm and lmPose as representative of the following methods.
        lm = keypoints.pose_landmarks
        lmPose  = mp_pose.PoseLandmark

        if lm is not None:
            # Left shoulder.
            l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
            l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)

            # Right shoulder.
            r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
            r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)

            # Left ear.
            l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
            l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)

            # Left hip.
            l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
            l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

            # Calculate distance between left shoulder and right shoulder points.
            offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)

            # Assist to align the camera to point at the side view of the person.
            # Offset threshold 30 is based on results obtained from analysis over 100 samples.
            if offset < 100:
                cv2.putText(image, str(int(offset)) + ' Aligned', (w - 200, 30), FONT, 0.6, GREEN, 2)
            else:
                cv2.putText(image, str(int(offset)) + ' Not Aligned', (w - 200, 30), FONT, 0.6, RED, 2)

            # Calculate angles.
            neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
            torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

            # Draw landmarks.
            cv2.circle(image, (l_shldr_x, l_shldr_y), 7, YELLOW, -1)
            cv2.circle(image, (l_ear_x, l_ear_y), 7, YELLOW, -1)

            # Let's take y - coordinate of P3 100px above x1,  for display elegance.
            # Although we are taking y = 0 while calculating angle between P1,P2,P3.
            cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, YELLOW, -1)
            cv2.circle(image, (r_shldr_x, r_shldr_y), 7, PINK, -1)
            cv2.circle(image, (l_hip_x, l_hip_y), 7, YELLOW, -1)

            # Similarly, here we are taking y - coordinate 100px above x1. Note that
            # you can take any value for y, not necessarily 100 or 200 pixels.
            cv2.circle(image, (l_hip_x, l_hip_y - 100), 7, YELLOW, -1)

            # Put text, Posture and angle inclination.
            # Text string for display.
            angle_text_string = 'Neck : ' + str(int(neck_inclination)) + '  Torso : ' + str(int(torso_inclination))

            # Determine whether good posture or bad posture.
            # The threshold angles have been set based on intuition.
            if neck_inclination < 40 and torso_inclination < 10:        
                cv2.putText(image, angle_text_string, (10, 30), FONT, 0.9, LIGHT_GREEN, 2)
                cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), FONT, 0.9, LIGHT_GREEN, 2)
                cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), FONT, 0.9, LIGHT_GREEN, 2)

                # Join landmarks.
                cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), GREEN, 4)
                cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), GREEN, 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), GREEN, 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), GREEN, 4)

            else:
                cv2.putText(image, angle_text_string, (10, 30), FONT, 0.9, RED, 2)
                cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), FONT, 0.9, RED, 2)
                cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), FONT, 0.9, RED, 2)

                # Join landmarks.
                cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), RED, 4)
                cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), RED, 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), RED, 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), RED, 4)

        cv2.imshow('Result', image)
        if cv2.waitKey(10) == 27:
            break
