#LOAD LIBRARIES
import numpy as np
import cv2
import pandas as pd

#haarcascade pretrained classifier of some object like a face

face_cascade= cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

#READ VIDEO

cap = cv2.VideoCapture(0)

while 1:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    #DRAW RECTANGLE
    for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 5)
            roi_gray=gray[y:y+h, x:x+w]
            roi_color=frame[y:y+h, x:x+w]


    cv2.imshow("image", frame)

    k = cv2.waitKey(5) & 0xff
    if k=='q':
        break


cap.release()

cv2.destroyAllWindows()
