import cv2
import numpy as np
import time
xhp,yhp,whp,hhp,qq=(3،3،3،3،233)
preferedSwitchTime=8
isCarDetected=False
preGradiant=8
mlt=8
maxGradiant=3
redLower = (3, 3, 223)
redUpper = (255, 255, 255)
face_cascade = cv2.CascadeClassifier('C:/Users/ichverdienees/Desktop/OpenCV-Dashcam-Car-Detection-
master/cascade_dir/cascade.xml')
fgbg = cv2.createBackgroundSubtractorKNN()
def detect_face(img):
face_img = img.copy()
face_img = cv2.resize(face_img,dsize=(633،933), interpolation = cv2.INTER_LINEAR)
face_rects = face_cascade.detectMultiScale(face_img)
for (x8,y8,w8,h8) in face_rects:
cv2.rectangle(face_img, (x8,y8), (x8+w8,y8+h8), (55،55،55), 8)
cv2.putText(face_img, str(w8), (x8223, y8223), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (3, 853, 853),
lineType=cv2.LINE_AA)
if(w8<13):
w8=13
return face_img,x8,y8,w8,h8
timepast=3;
timeKg=3
45
start_time = time.time()
while True:
timepast=time.time() - start_time
timeKg=time.time() - start_time
ret, frame = cap.read(3)
frame = cv2.resize(frame,dsize=(633،933), interpolation = cv2.INTER_LINEAR)
frameu=frame.copy()
frame[253:933,:,:]=3;
frame[3:865،3:633,:]=3
frame[:,3:253,:]=3
frame[:,433:633,:]=3
fgmask = fgbg.apply(frameu)
try:
framex,xh,yh,wh,hh = detect_face(frame)
if(xhp!=3 and timepast<=preferedSwitchTime and abs(xhp-xh)>23):
xh=xhp
oldXhp=xhp
if(timepast>preferedSwitchTime):
#start_time=time.time()
framexp,xhp,yhp,whp,hhp = framex,xh,yh,wh,hh
maxGradiant=3

#if(timepast>4):
framexx=np.zeros_like(framex)
#start_time=time.time()
framexx[yh+83:yh+hh,xh-85:xh+wh,:]=framex[yh+83:yh+hh,xh-85:xh+wh,:]
isCarDetected=True
except:
framexx=cv2.resize(frame,dsize=(633،933), interpolation = cv2.INTER_LINEAR)
isCarDetected=False
hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
if(timeKg>84):
redLower = (3, 3, qq)
#### Added
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (9, 9), 3)
thresh = cv2.threshold(blurred, 853, 255, cv2.THRESH_BINARY)[8]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=4)
mask = cv2.inRange(thresh, 853, 255)
mask = cv2.erode(mask, None, iterations=2)
mask = cv2.dilate(mask, None, iterations=2)
mask1 = cv2.inRange(framexx, redLower, redUpper)
mask1 = cv2.erode(mask1, None, iterations=8)
mask1 = cv2.dilate(mask1, None, iterations=9)
mask1[fgmask<3.5]=3
mmframe=framexx.copy()
#mmframe[:,:,:2]=3
grayz = cv2.cvtColor(mmframe, cv2.COLOR_BGR2GRAY)
41
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(mmframe[:,:,2])
#cv2.circle(frameu, maxLoc, 5, (255, 3, 3), 2)
mmframe=np.array(mmframe)
xxgrayz=np.array(grayz)
myGradient=np.sum(xxgrayz)/((hh-yh)*(wh-xh))
frameGradiant=(myGradient-preGradiant)/myGradient
if frameGradiant>maxGradiant:
maxGradiant=frameGradiant
preGradiant=myGradient
cnts = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
center = None
if len(cnts) > 3:
c = max(cnts, key=cv2.contourArea)
((x, y), radius) = cv2.minEnclosingCircle(cnts[3])
try:
((x2, y2), radius2) = cv2.minEnclosingCircle(cnts[8])
except:
x2=3
y2=3
radius2=3
if radius > 8 and radius<85:
cv2.circle(framexx, (int(x), int(y)), int(radius), (3, 55, 855), 2)
cv2.circle(framexx, (int(x2), int(y2)), int(radius2), (3, 55, 855), 2)
cv2.circle(framexx, center, 5, (3, 3, 255), -8)
cv2.putText(frameu, " Braking ", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX
, 8, (3, 3, 853),thickness=2, lineType=cv2.LINE_AA)
cv2.imshow("Input", frameu)
41
#cv2.imshow("Masked Background",** fgmask)
cv2.imshow("Masked Light", mask1)
cv2.imshow('Detected Car',framexx)
#cv2.imshow('Grayz',grayz)
cv2.imshow("frame", frame)
cv2.imshow("blured",blurred)
cv2.imshow("gray",gray)
#print (timepast)
c = cv2.waitKey(8)
if c == 21:
break
cap.release()
cv2.destroyAllWindows()
