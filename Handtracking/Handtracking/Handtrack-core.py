import cv2
import mediapipe as mp
import time


mpHands = mp.solutions.hands
hands = mpHands.Hands(False)
mpDraw = mp.solutions.drawing_utils

pTime=0
cTime=0

cap = cv2.VideoCapture(0)

while True:

    ReadOK, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)


    if results.multi_hand_landmarks:
        for handslm in results.multi_hand_landmarks:

            for id, lm in enumerate(handslm.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                #if id == 12 :
                cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handslm, mpHands.HAND_CONNECTIONS)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)

    cv2.imshow('img', img)


    cv2.waitKey(1)
