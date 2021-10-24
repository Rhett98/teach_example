import cv2

src = cv2.imread("./pic/test.png")

dst = cv2.bilateralFilter(src, 9, 75, 75)

cv2.imshow("src", src)
cv2.imshow("dst", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()