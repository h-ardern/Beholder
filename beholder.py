from face_detect import face_rec
from face_capture import add_person
from sys import platform
from trainer import train
import os
import cv2

# Path for face image database
path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")


def splash():
    s = open('Splash.txt', 'r', encoding='utf-8')
    splash_screen = s.read()
    print(splash_screen)
    s.close()


def init():
    option = int(input(
        'BEHOLDER _ INPUT//: Please press 1 to add a person to the dataset \n                    Press 2 to run the system or 3 to quit: '))
    if option == 1:
        add_usr()
        init()
    elif option == 2:
        face_rec(camera_select())
    elif option == 3:
        exit()
    else:
        print('Not a valid input')
        init()


def add_usr():
    add_person()
    train(path, recognizer, detector)


def camera_select():
    cam = input(
        "BEHOLDER _ INPUT//: PLEASE ENTER ID OF DESIRED CAMERA\nPRESS ENTER FOR DEFAULT | ENTER \'L\' FOR LIST OF DEVICES} - ")
    if cam == "":
        cam = 0
    elif cam == 'L' or cam == 'l':
        if platform == "darwin":
            os.system('ioreg | grep -i cam')
            cam = camera_select()
            return cam
        elif platform == "win32" or platform == "win64":
            pass
    else:
        try:
            cam = int(cam)
        except:
            print("ERROR NOT A VALID DEVICE")
    return cam


splash()
init()
