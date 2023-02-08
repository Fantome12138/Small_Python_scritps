import cv2
import numpy as np

'''经常用到轮廓查找和多边形拟合等opencv操作，因此记录以备后续使用。本文代码中的阈值条件对图片没有实际意义，仅仅是为了测试。'''

img = cv2.imread('/home/yasin/coffe.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, contours, -1, (255, 0, 255), 1)
cv2.namedWindow('Result of drawContours', 0)
cv2.imshow('Result of drawContours', img)
cv2.waitKey()


cnt = 0
for i in range(len(contours)):
    arclen = cv2.arcLength(contours[i], True)
    epsilon = max(3, int(arclen * 0.02))   # 拟合出的多边形与原轮廓最大距离，可以自己设置，这里根据轮廓周长动态设置
    approx = cv2.approxPolyDP(contours[i], epsilon, False) # 轮廓的多边形拟合
    area = cv2.contourArea(contours[i])     # 计算面积
    rect = cv2.minAreaRect(contours[i])
    box = np.int0(cv2.boxPoints(rect))      # 计算最小外接矩形顶点
    h = int(rect[1][0])
    w = int(rect[1][1])
    if min(h, w) == 0:
        ration = 0
    else:
        ration = max(h,w) /min(h,w)   # 长宽比

    # 对长宽比，轮廓面积，拟合出的多边形顶点数做筛选
    if ration < 10 and area > 20 and area < 4000 and approx.shape[0] > 3 :
        # 对满足条件的轮廓画出轮廓的拟合多边形
        cv2.polylines(img, [approx], True, (0, 255, 0), 1)

cv2.namedWindow('Result of filtered', 0)
cv2.imshow('Result of filtered', img)
cv2.waitKey()

