import cv2
import mediapipe as mp
import os
import time

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

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Corrected the typo here
        results = pose.process(imgRGB)
        print(results.pose_landmarks)
        if results.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks,mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape
                print(id, lm)
                cx, cy = int(lm.x * w) ,int (lm.y * h)
                cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)

        cv2.imshow("Image", img) 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





