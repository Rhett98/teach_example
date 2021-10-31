from csk import *
import cv2 
import time

sequence_path = "../video/ex1.mp4"
cap = cv2.VideoCapture(sequence_path)
# 视频的第一帧
ret, img=cap.read()
frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 设置窗口的初始位置
x, y, w, h = 300, 200, 100, 50 # simply hardcoded the values

tracker = CSK() # CSK instance
tracker.init(frame,x,y,w,h) # initialize CSK tracker with GT bounding box

while True:
    ret, img=cap.read()
    if ret == True:
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        t1 = time.time()
        x, y = tracker.update(frame) # update CSK tracker and output estimated position
        t2 = time.time()
        print('spend time:',t2-t1,'s')
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.imshow("frame", img)
        k=cv2.waitKey(30) & 0xFF
        if k==27:
            break
    else:
        break
# clean up the camera and close any open windows
cap.release()
cv2.destroyAllWindows()
