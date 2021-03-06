import cv2 as cv
img = cv.imread("./pic/home.png")

surf = cv.xfeatures2d.SURF_create(7000)
kp, des = surf.detectAndCompute(img,None)
img2 = cv.drawKeypoints(img,kp,None,(255,0,0))

cv.imshow('surf', img2)
cv.waitKey(0)
cv.destroyAllWindows()
