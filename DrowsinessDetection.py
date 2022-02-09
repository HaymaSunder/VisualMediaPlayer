import cv2
import dlib
import vlc
from scipy.spatial import distance
import ctypes
from ctypes import *
import winsound

Instance = vlc.Instance()
player = Instance.media_player_new()
path = Instance.media_new('D:/SEM 6/PROJECTS/Gesture-Orientation/media/Vaathi Coming.mp4')
player.set_media(path)
player.play()

NORM_FONT = ("Verdana", 14)
def Mbox(title, text, style):
    sty=int(style)+ 4096
    return windll.user32.MessageBoxW(0, text, title, sty)

def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A+B)/(2.0*C)
    return ear_aspect_ratio

cap = cv2.VideoCapture(1)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("D:/SEM 6/PROJECTS/Gesture-Orientation/shape_predictor_68_face_landmarks.dat")

counter=0
force_retry=0
flag=0
while flag!=1 and force_retry!=3:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:
        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []

        for n in range(36,42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x,y))
            next_point = n+1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

        for n in range(42,48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x,y))
            next_point = n+1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)

        EAR = (left_ear+right_ear)/2
        EAR = round(EAR,2)
        if EAR<0.23:
            counter+=1
            if(counter>10):
                player.pause()
                print("Drowsy")
                winsound.PlaySound('C:/Users/LENOVO/Desktop/Gesture-Orientation/windows_error.wav', winsound.SND_ASYNC)
                response = Mbox('Drowsiness Detected', 'Do you want to continue watching?', 4)
                if response == 6:
                    print("Yes Clicked")
                    counter=0
                    force_retry+=1
                    player.play()
                elif response == 7:
                    print("No Clicked")
                    player.stop()
                    flag=1
                    break
        else: 
            counter=0
        print(EAR)

    cv2.imshow("Are you Sleepy", frame)

    if cv2.waitKey(10) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()