import cv2 as cv
import matplotlib.pyplot as plt
img1 = cv.imread('C:/Users/dell/Desktop/mayun1.png')
img2 = cv.imread('C:/Users/dell/Desktop/mayun2.png')
img1 = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
img2 = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)
orb = cv.ORB_create()

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)

matches = bf.match(des1, des2)

matches = sorted(matches, key = lambda x:x.distance)

img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[: 20], None, flags=2)
plt.imshow(img3), plt.show()
