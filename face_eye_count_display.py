import cv2 as cv
import numpy as np


cap = cv.VideoCapture(0)
 
# Check if the webcam is opened correctly 
if not cap.isOpened(): 
    raise IOError("Cannot open webcam") 
 
# detector=cv.CascadeClassifier("C:\\OpenCV\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml")
# eye_cascade = cv.CascadeClassifier('C:\\OpenCV\\opencv\\sources\\data\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml')
eye_cascade=cv.CascadeClassifier("C:\\Users\\Mohit\\Anaconda3\\envs\\ML_LCO\\Lib\\site-packages\\cv2\\data\\haarcascade_eye_tree_eyeglasses.xml")
detector=cv.CascadeClassifier("C:\\Users\\Mohit\\Anaconda3\\envs\\ML_LCO\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")
while True: 
    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    
    faces=detector.detectMultiScale(frame,1.3,5)
    
    if(len(faces) > 0):
        
        for face in faces:
    
            x,y,w,h=face

            cv.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),5)

            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)

            if len(eyes) > 0:
                
                print(str(len(faces)) + " Faces, " + str(len(eyes)) + " Eye detected!")
                for (x_eye,y_eye,w_eye,h_eye) in eyes: 
                    center = (int(x_eye + 0.5*w_eye), int(y_eye + 0.5*h_eye)) 
                    radius = int(0.3 * (w_eye + h_eye)) 
                    color = (0, 255, 0) 
                    thickness = 3 
                    cv.circle(roi_color, center, radius, color, thickness)

            else:
                print(str(len(faces)) + " Faces, " + "No eyes Found")

    else:
        print("No Face!")
    
    cv.imshow('Input', frame) 
 
    c = cv.waitKey(1) 
    if c == 27: 
        break 
 
cap.release() 
cv.destroyAllWindows()