import cv2
import numpy as np

# 打开两个文件
img1 = cv2.imread("D:/keypoint/orig.png")
#img2 = cv2.imread("H:/work/underwater-uieb/data/test/test1_result/862-1.png")
#img2 = cv2.imread("H:/work/underwater-uieb/data/add_dropout/test/865-1.png")
#img2 = cv2.imread("H:/work/single/Underwater Image Enhancement/HE/OutputImages/862-1.png")
#img2 = cv2.imread("H:/work/single/Underwater Image Enhancement/CLAHE/OutputImages/862-1.png")
img2 = cv2.imread("D:/keypoint/orig1.png")

# 灰度化
g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 他建SIFT特征检测器
sift = cv2.xfeatures2d.SIFT_create()

# 计算描述子与特征点
kp1, des1 = sift.detectAndCompute(g1, None)
kp2, des2 = sift.detectAndCompute(g2, None)

# 创建匹配器
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# 对描述子进行匹配计算
matchs = flann.knnMatch(des1, des2, k=2)

good = []
for i, (m, n) in enumerate(matchs):
    if m.distance < 0.7 * n.distance:
        good.append(m)

if len(good) >= 4:
    srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # 查找单应性矩阵
    H, _ = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5.0)

    h, w = img1.shape[:2]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H)

    cv2.polylines(img2, [np.int32(dst)], True, (0, 0, 255))
else:
    print('the number of good is less than 4.')
    exit()

ret = cv2.drawMatchesKnn(img1, kp1, img2, kp2, [good], None)
print('特征的匹配个数为：',len(good))
cv2.imshow('result', ret)
cv2.waitKey(0)
cv2.destroyAllWindows()
