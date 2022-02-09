import cv2
import numpy as np
import math
import vlc
import time
import dlib
from ctypes import *
import ctypes
from scipy.spatial import distance
import winsound

Instance = vlc.Instance()
player = Instance.media_player_new()
path = Instance.media_new('D:/SEM 6/PROJECTS/Gesture-Orientation/media/Vaathi Coming.mkv')
player.set_media(path)

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

cap = cv2.VideoCapture(0)

hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("D:/SEM 6/PROJECTS/Gesture-Orientation/shape_predictor_68_face_landmarks.dat")

counter=0
force_retry=0
flag=0

counter_pause=0
counter_play=0
counter_stop=0
counter_volume=0
counter_title=0
factor=10


while flag!=1 and force_retry!=3:
        
    try:  #an error comes if it does not find anything in window as it cannot find contour of max area
          #therefore this try error statement
          
        ret, frame = cap.read()
        frame=cv2.flip(frame,1)

        kernel = np.ones((3,3),np.uint8)

        #define region of interest
        roi=frame[100:300, 400:600]        
        
        cv2.rectangle(frame,(400,100),(600,300),(0,255,0),0)    
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # define range of skin color in HSV
        lower_skin = np.array([0,20,70], dtype=np.uint8)
        upper_skin = np.array([20,255,255], dtype=np.uint8)
        
     #extract skin colour image 
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
    #extrapolate the hand to fill dark spots within
        mask = cv2.dilate(mask,kernel,iterations = 4)
        
    #blur the image
        mask = cv2.GaussianBlur(mask,(5,5),100) 

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
                if(counter>50):
                    player.pause()
                    print("Drowsy")
                    winsound.PlaySound('D:/SEM 6/PROJECTS/Gesture-Orientation/windows_error.wav', winsound.SND_ASYNC)
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
        cv2.imshow("Drowsiness Detector",frame)
        
        
        
         
       
    #find contours
        contours,hierarchy= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
   #find contour of max area(hand)
        cnt = max(contours, key = lambda x: cv2.contourArea(x))
        
    #approx the contour a little
        epsilon = 0.0005*cv2.arcLength(cnt,True)
        approx= cv2.approxPolyDP(cnt,epsilon,True)
       
        
    #make convex hull around hand
        hull = cv2.convexHull(cnt)
        
     #define area of hull and area of hand
        areahull = cv2.contourArea(hull)
        areacnt = cv2.contourArea(cnt)
      
    #find the percentage of area not covered by hand in convex hull
        arearatio=((areahull-areacnt)/areacnt)*100
    
     #find the defects in convex hull with respect to hand
        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)
        
    # l = no. of defects
        l=0
        
    #code for finding no. of defects due to fingers
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            pt= (100,180)
            
            
            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            s = (a+b+c)/2
            ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
            
            #distance between point and convex hull
            d=(2*ar)/a
            
            # apply cosine rule here
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            
        
            # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
            if angle<=90 and d>30:
                l += 1
                cv2.circle(roi, far, 3, [255,0,0], -1)
            
            #draw lines around hand
            cv2.line(roi,start, end, [0,255,0], 2)
            
#************************************************************************************************************************
        l+=1
        
        #print corresponding gestures which are in their ranges
        font = cv2.FONT_HERSHEY_SIMPLEX
        if l==1:
            if areacnt<2000:
                cv2.putText(frame,'Put hand in the box',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            else:
                counter_play=0
                counter_stop=0
                counter_volume=0
                counter_title=0
                if arearatio<25:
                    # cv2.putText(frame,'0',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA) 
                    counter_pause+=1
                    if counter_pause>=3:
                        player.set_pause(1)
                        counter_pause=0                
                # else:
                #     cv2.putText(frame,'1',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        elif l==2:
            counter_pause=0
            counter_play=0
            counter_stop=0
            counter_title=0
            counter_volume+=1
            if counter_volume>=2:
                factor=factor+5
                player.audio_set_volume(factor)
                counter_volume=0
            cv2.putText(frame,'2',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        elif l==3:
            counter_pause=0
            counter_play=0
            counter_volume=0
            counter_title=0
            cv2.putText(frame,'3',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            counter_stop+=1
            if counter_stop>=5:
                player.stop()
                counter_stop=0                 
        elif l==4:
            counter_pause=0
            counter_play=0
            counter_stop=0
            counter_title=0
            counter_volume+=1
            if counter_volume>=2:
                factor=factor-5
                player.audio_set_volume(factor)
                counter_volume=0
            cv2.putText(frame,'4',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        elif l==5:
            counter_pause=0
            counter_title=0
            counter_stop=0
            counter_volume=0
            counter_play+=1
            if counter_play>=5:
                player.play()
                counter_play=0 
            cv2.putText(frame,'5',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)            
            
        #show the windows
        cv2.imshow('mask',mask)
        cv2.imshow('Visual Media Player',frame)
    except:
        pass
        
    if cv2.waitKey(10) == ord('q'):
        break
    
cv2.destroyAllWindows()
cap.release()   