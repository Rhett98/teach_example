import cv2

src = cv2.imread("./pic/Lena.png")

dst = cv2.GaussianBlur(src, (7, 7), 0)

cv2.imshow("src", src)
cv2.imshow("dst", dst)

cv2.waitKey(0)
cv2.destroyAllwindows()