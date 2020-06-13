import numpy as np
import matplotlib.pyplot as plt
import cv2
from model import FacialExpressionModel
from PIL import Image

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

def detect_face(img):
    
    face_img = img.copy()

    gray_fr = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_fr,scaleFactor=1.2,minNeighbors=5)

    for (x,y,w,h) in faces:
        fc = gray_fr[y:y + h, x:x + w]
        roi = cv2.resize(fc, (48, 48))
        pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

        cv2.putText(face_img, pred, (30, 30), font, 1, (255, 255, 0), 2)

        img2 = cv2.imread("./Images/"+pred+".png")
        img3 = cv2.resize(img2, face_img[y:y+h, x:x+w].shape[0:2])

        face_img[y:y + h, x:x + w] = img3

   
    return face_img

cap = cv2.VideoCapture(0)

while True:
    
    ret,frame = cap.read(0)
    
    frame = detect_face(frame)
    
    cv2.imshow('Video Face Detect',frame)
    
    k = cv2.waitKey(1)
    
    if k == 27:
        break
        
cap.release()
cv2.destroyAllWindows()