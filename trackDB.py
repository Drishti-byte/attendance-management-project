import cv2 
import numpy as np
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('TrainFile/Trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
cam = cv2.VideoCapture(0)
#font = cv2.InitFont(cv2.CV_FONT-HERSHEY_SIMPLEX, 1, 1, 0, 1, 1)
while True: 
    ret, im = cam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im, (x,y), (x+w, y+h), (255, 0, 0), 2)
        Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
        if(conf < 50): 
            if(Id == 1):
                Id = "Drishti"
            elif(Id == 7):
                Id = "Drishti"
            else:
                Id = "Unknown"
        cv2.putText(im, str(Id), (x, y+h),
                    cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))
        # cv2.cv.putText(cv2.cv.fromarray(im), str(Id), (x, y+h), font, 255)
    cv2.imshow('im', im)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break 
cam.release()
cv2.destroyAllWindows()