from cProfile import label
import cv2 as cv
import numpy as np
import os 

DIR =r'C:\Users\Harish\Documents\face recognition\data\train'
people =[]
for person in os.listdir(DIR):
    people.append(person)

features = []
labels = []


def pre_train():
    for person in os.listdir(DIR):
        path = os.path.join(DIR,person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            image_ = cv.imread(img_path)
            gray = cv.cvtColor(image_,cv.COLOR_BGR2GRAY)

            haar = cv.CascadeClassifier('haar.xml')
            face_recr = haar.detectMultiScale(gray,1.1,4)


            for (x,y,w,h) in face_recr:
                face_roi = gray[y:y+h,x:x+h]
                features.append(face_roi)
                labels.append(label)

pre_train()

face_recog = cv.face.LBPHFaceRecognizer_create()

features = np.array(features,dtype='object')
labels = np.array(labels)

face_recog.train(features,labels)
face_recog.save('face_recog.yml')
