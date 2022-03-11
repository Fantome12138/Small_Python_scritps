'''
ORB (Oriented FAST and Rotated BRIEF)算法
分为两部分：
1\特征点提取 -由FAST（Features from Accelerated Segment Test）算法发展来的
2\特征点描述 -根据BRIEF（Binary Robust IndependentElementary Features）特征描述算法改进的

AKAZE算法
'''
import numpy as np
import cv2


def drawMatches(img1, kp1, img2, kp2, matches):
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    out = np.zeros((max([rows1,rows2]),cols1 + cols2, 3),dtype = 'uint8')
    # 拼接图像
    out[:rows1, :cols1] = np.dstack([img1, img1,img1])
    out[:rows2, cols1:] = np.dstack([img2, img2,img2])

    for mat in matches:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt
        # 绘制匹配点
        cv2.circle(out, (int(x1),int(y1)),4,(255,255,0),1)
        cv2.circle(out,(int(x2)+cols1,int(y2)),4,(0,255,255),1)
        cv2.line(out,(int(x1),int(y1)),(int(x2)+cols1,int(y2)),(255,0,0),1)
    return out
    
img1 = cv2.imread("data/face1.jpg", 0)  # 导入灰度图像
img2 = cv2.imread("data/face2.jpg", 0)

detector = cv2.ORB_create()
detector = cv2.AKAZE_create()  # 两种算法

kp1 = detector.detect(img1, None)
kp2 = detector.detect(img2, None)
kp1, des1 = detector.compute(img1, kp1)
kp2, des2 = detector.compute(img2, kp2)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
matches = bf.match(des1, des2)
img3 = drawMatches(img1, kp1, img2, kp2, matches[:50])
# img3 = cv2.drawKeypoints(img1,kp,None,color = (0,255,0),flags = 0)
cv2.imwrite("orbTest.jpg",img3)
cv2.imshow('orbTest',img3)
cv2.waitKey(0)






