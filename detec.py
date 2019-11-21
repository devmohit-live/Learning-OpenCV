import numpy as np,cv2 as cv
eyedetector=cv.CascadeClassifier("C:\\Users\\Mohit\\Anaconda3\\envs\\ML_LCO\\Lib\\site-packages\\cv2\\data\\haarcascade_eye.xml")
facedetector=cv.CascadeClassifier("C:\\Users\\Mohit\\Anaconda3\\envs\\ML_LCO\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")
cap = cv.VideoCapture(0)
 
if not cap.isOpened(): 
    raise IOError("Cannot open webcam") 
while True: 
    ret, frame = cap.read()   
    faces=facedetector.detectMultiScale(frame,1.3,5)
    eyes=facedetector.detectMultiScale(frame,1.3,5)

    if(len(faces) > 0):
        
        for face in faces:
    
            x,y,w,h=face

            cv.rectangle(frame,(x,y),(x+w,y+h),(100,100,150),5)

        for eye in eyes:
            x,y,w,h=face

            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),5)
 
    cv.imshow('Input', frame) 
 
    c = cv.waitKey(1) 
    if c == 27: 
        break 
cap.release() 
cv.destroyAllWindows()