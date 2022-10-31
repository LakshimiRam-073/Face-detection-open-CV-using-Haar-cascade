from symbol import factor
import cv2 as cv

image = cv.imread(r"C:\Users\Harish\Pictures\Camera Roll\images.jpg")
cv.imshow('me',image)

gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

haar_cascade = cv.CascadeClassifier('haar.xml')

face_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=1)

print(len(face_rect))

for (x,y,w,h) in face_rect:
    cv.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

cv.imshow('face detected',image)
print("face detected")
cv.waitKey(0)