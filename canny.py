# coding=utf-8
import cv2
import numpy as np

#img = cv2.imread("H:/work/underwater-uieb/data/add_dropout/test/874.png", 0)
#img = cv2.imread("H:/work/underwater-uieb/data/test/testup_result/874.png", 0)
img = cv2.imread("D:/candy/RAY.png", 0)
#img = cv2.imread("H:/work/single/Underwater Image Enhancement/HE/OutputImages/874.png", 0)

img = cv2.GaussianBlur(img, (3, 3), 0)  # 用高斯平滑处理原图像降噪。
canny = cv2.Canny(img, 50, 150)  # 最大最小阈值

cv2.imshow('Canny', canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("D:/candy/RAY1.png", canny)