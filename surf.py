import cv2 as cv
img = cv.imread("C:/Users/dell/Desktop/lena512color.png")

surf = cv.xfeatures2d.SURF_create(7000)

kp, des = surf.detectAndCompute(img,None)

img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)
cv.imshow('surf', img2)

cv.waitKey(0)
cv.destroyAllWindows()
