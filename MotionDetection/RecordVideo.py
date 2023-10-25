import numpy as np
import cv2

project_directory = "C:/Users/corne/OneDrive - Technological University Dublin/FinalYearProject/PoseVideos"

cap = cv2.VideoCapture(0)


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(project_directory + "/3.mp4.mov", fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        out.write(frame)
        cv2.imshow('output', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
