import cv2
import mediapipe as mp
import os
import time
import math

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

project_directory = "C:/Users/corne/OneDrive - Technological University Dublin/FinalYearProject"
video_file_path = os.path.join(project_directory, "PoseVideos", "3.mp4.mov")

cap = cv2.VideoCapture(video_file_path)
pTime = 0

while True:
    success, img = cap.read()
    
    if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # Display the FPS on the image
        cv2.putText(img, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        results = pose.process(imgRGB)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Extract the x-coordinates of points 23, 24, 25, and 26
            x23, x24, x25, x26 = landmarks[23].x, landmarks[24].x, landmarks[25].x, landmarks[26].x

            # Calculate the angle between lines 23-24 and 25-26
            angle_threshold = 60  # 60 degrees tolerance
            angle_23_24_25_26 = math.degrees(math.atan(abs((x25 - x24) / (landmarks[25].y - landmarks[24].y))) - math.atan(abs((x23 - x24) / (landmarks[23].y - landmarks[24].y)))
                                           + math.atan(abs((x26 - x25) / (landmarks[26].y - landmarks[25].y))) - math.atan(abs((x24 - x25) / (landmarks[24].y - landmarks[25].y))))
            
            if abs(angle_23_24_25_26) < angle_threshold:
                # Draw the lines in green
                pt1 = (int(x23 * img.shape[1]), int(landmarks[23].y * img.shape[0]))
                pt2 = (int(x24 * img.shape[1]), int(landmarks[24].y * img.shape[0]))
                cv2.line(img, pt1, pt2, (0, 255, 0), 3)

                pt1 = (int(x24 * img.shape[1]), int(landmarks[24].y * img.shape[0]))
                pt2 = (int(x25 * img.shape[1]), int(landmarks[25].y * img.shape[0]))
                cv2.line(img, pt1, pt2, (0, 255, 0), 3)

                pt1 = (int(x25 * img.shape[1]), int(landmarks[25].y * img.shape[0]))
                pt2 = (int(x26 * img.shape[1]), int(landmarks[26].y * img.shape[0]))
                cv2.line(img, pt1, pt2, (0, 255, 0), 3)

            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

