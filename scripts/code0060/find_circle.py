'''
1\ opencv 斑点(Blob)检测--SimpleBlobDetector_create
https://blog.csdn.net/u014072827/article/details/111033547
'''
import cv2
import numpy as np
 
# Read image
im = cv2.imread("./1.jpg", cv2.IMREAD_GRAYSCALE)
 
# 设置SimpleBlobDetector_Params参数
params = cv2.SimpleBlobDetector_Params()
# 改变阈值
params.minThreshold = 10
params.maxThreshold = 200
# 通过面积滤波
params.filterByArea = True
params.minArea = 1500
# 通过圆度滤波
params.filterByCircularity = True
params.minCircularity = 0.1
# 通过凸度滤波
params.filterByConvexity = True
params.minConvexity = 0.87
# 通过惯性比滤波
params.filterByInertia = True
params.minInertiaRatio = 0.01
# 创建一个检测器并使用默认参数
detector = cv2.SimpleBlobDetector_create(params)
# 检测blobs.
key_points = detector.detect(im)
 
# 绘制blob的红点
draw_image = cv2.drawKeypoints(im, key_points, np.array([]), (0, 0, 255),
                               cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite('./plog.jpg', draw_image)


'''
2\ cv2.HoughCircles
'''
import cv2
import numpy as np
img = cv2.imread('./1.jpg')
img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
circles = cv2.HoughCircles(img , cv2.HOUGH_GRADIENT , 1 , 90 , param1=100 , param2= 30 , minRadius=30 , maxRadius= 50)

if len(circles[0]) < 1:
    print("waring: 检测到的角点数量不足！")
else:
    # 输出检测到圆的个数
    print("检测到圆的个数：")
    print(len(circles[0]))
    print(circles)
    print('circles[0]):\n',circles[0])
print('------------------------------')
for circle in circles[0]:
    x = int(circle[0])
    y = int(circle[1])
    r = int(circle[2])
    # print(circle[0])
    # print('------------------------')
    # print(circle[1])
    # print('-------------------------')
    # print(circle[2])
    # print('--------------------------')
    draw_circle = cv2.circle(img ,(x,y) ,r ,(255,255,255) ,1,10 ,0) #画出检测到的圆，（255,255,255）代表白色

circles_order = [None] * 4
x_list = []
y_list = []
for i in circles[0]:
    x_list.append(i[0])
    y_list.append(i[1])
print(x_list)
print('----------')
print(y_list)
print('-----------')
center = (np.mean(x_list), np.mean(y_list))
print(center)

# 顺时针调整角点坐标
for i in circles[0]:
    if i[1] < center[1]:
        if i[0] < center[0]:
            circles_order[0] = list(i[:-1])
        else:
             circles_order[1] = list(i[:-1])
    else:
        if i[0] > center[0]:
            circles_order[2] = list(i[:-1])
        else:
            circles_order[3] = list(i[:-1])

cv2.imwrite('./plot.jpg', draw_circle)
