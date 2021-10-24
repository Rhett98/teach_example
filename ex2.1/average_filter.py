import cv2

src = cv2.imread("./pic/Lena.png")
dst = cv2.blur(src, (9, 9))

cv2.imshow("src", src)
cv2.imshow("dst", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()