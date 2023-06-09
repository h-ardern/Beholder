import cv2
import numpy as np
import os
import json


def face_rec(cam):
    print('BEHOLDER //: FACE_RECOGNITION LOADING')
    f = open('users/authorised.json', "rb")
    users = json.load(f)
    print(users)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    cascadePath = "Cascades/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # iniciate id counter
    id = 0
    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(cam)
    cam.set(3, 640)  # set video widht
    cam.set(4, 480)  # set video height
    # Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )
        for (x, y, w, h) in faces:
            color = (0, 255, 255)
            try:
                if users[str(id)] == True:
                    color = (0, 255 , 0)
                elif users[str(id)] == False:
                    color = (0, 0, 255)
            except:
                color = (0, 255, 255)

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            # If confidence is less them 100 ==> "0" : perfect match
            if (confidence < 100):
                id = id
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(
                img,
                str(id),
                (x + 5, y - 5),
                font,
                1,
                (255, 255, 255),
                2
            )
            cv2.putText(
                img,
                str(confidence),
                (x + 5, y + h - 5),
                font,
                1,
                (255, 255, 0),
                1
            )

        cv2.imshow('camera', img)
        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()
