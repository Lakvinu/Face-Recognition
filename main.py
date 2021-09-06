import numpy
import face_recognition as fr
import cv2

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)


while True:

    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for x, y, w, h in faces:

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 5)



    cv2.imshow('Face Recogniser', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()