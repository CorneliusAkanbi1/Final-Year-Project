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

def check_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if det == 0:
        return False  # Lines are parallel

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / det
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / det

    if 0 <= t <= 1 and 0 <= u <= 1:
        return True  # Lines intersect

    return False

cap = cv2.VideoCapture(video_file_path)
pTime = 0
rep_counter = 0  # Initialize the repetition counter
contact_flag = False  # Initialize the flag to detect contact

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
            angle_23_24_25_26 = math.degrees(
                math.atan(abs((x25 - x24) / (landmarks[25].y - landmarks[24].y))) - math.atan(
                    abs((x23 - x24) / (landmarks[23].y - landmarks[24].y))) + math.atan(
                    abs((x26 - x25) / (landmarks[26].y - landmarks[25].y))) - math.atan(
                    abs((x24 - x25) / (landmarks[24].y - landmarks[25].y))))

            if abs(angle_23_24_25_26) < angle_threshold:
                # Check if the green lines come into contact
                line1 = (x23 * img.shape[1], landmarks[23].y * img.shape[0], x24 * img.shape[1], landmarks[24].y * img.shape[0])
                line2 = (x25 * img.shape[1], landmarks[25].y * img.shape[0], x26 * img.shape[1], landmarks[26].y * img.shape[0])

                if check_intersection(line1, line2):
                    if not contact_flag:
                        rep_counter += 1  # Increment the repetition counter
                        contact_flag = True

                if contact_flag:
                    color = (255, 0, 0)  # Turn the lines blue
                else:
                    color = (0, 255, 0)  # Lines remain green

                pt1 = (int(x23 * img.shape[1]), int(landmarks[23].y * img.shape[0]))
                pt2 = (int(x24 * img.shape[1]), int(landmarks[24].y * img.shape[0]))
                cv2.line(img, pt1, pt2, color, 3)

                pt1 = (int(x25 * img.shape[1]), int(landmarks[25].y * img.shape[0]))
                pt2 = (int(x26 * img.shape[1]), int(landmarks[26].y * img.shape[0]))
                cv2.line(img, pt1, pt2, color, 3)

            # Display the repetition count in the top-right corner
            cv2.putText(img, f"Reps: {rep_counter}", (img.shape[1] - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
