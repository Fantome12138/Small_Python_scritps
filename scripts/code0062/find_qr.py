import math
import cv2
import numpy as np

cnt = 3
cnt2 = 0

def showImg(img):
    # cv2.namedWindow('a', cv2.WINDOW_NORMAL)
    # cv2.imshow('a', img)
    # cv2.waitKey()
    global cnt,cnt2
    cnt2=cnt2+1
    path="test" + str(cnt)+"\\"+str(cnt2)+".jpg"
    cv2.imwrite(path, img)

def leftTop(centerPoint):
    minIndex = 0
    minMultiple = 10000

    multiple = (centerPoint[1][0] - centerPoint[0][0]) * (centerPoint[2][0] - centerPoint[0][0]) + (
            centerPoint[1][1] - centerPoint[0][1]) * (centerPoint[2][1] - centerPoint[0][1])
    if minMultiple > multiple:
        minIndex = 0
        minMultiple = multiple

    multiple = (centerPoint[0][0] - centerPoint[1][0]) * (centerPoint[2][0] - centerPoint[1][0]) + (
            centerPoint[0][1] - centerPoint[1][1]) * (centerPoint[2][1] - centerPoint[1][1])
    if minMultiple > multiple:
        minIndex = 1
        minMultiple = multiple

    multiple = (centerPoint[0][0] - centerPoint[2][0]) * (centerPoint[1][0] - centerPoint[2][0]) + (
            centerPoint[0][1] - centerPoint[2][1]) * (centerPoint[1][1] - centerPoint[2][1])
    if minMultiple > multiple:
        minIndex = 2

    return minIndex

def orderPoints(centerPoint, leftTopPointIndex):
    otherIndex = []
    waiji = (centerPoint[(leftTopPointIndex + 1) % 3][0] - centerPoint[(leftTopPointIndex) % 3][0]) * (
            centerPoint[(leftTopPointIndex + 2) % 3][1] - centerPoint[(leftTopPointIndex) % 3][1]) - (
                    centerPoint[(leftTopPointIndex + 2) % 3][0] - centerPoint[(leftTopPointIndex) % 3][0]) * (
                    centerPoint[(leftTopPointIndex + 1) % 3][1] - centerPoint[(leftTopPointIndex) % 3][1])
    if waiji > 0:
        otherIndex.append((leftTopPointIndex + 1) % 3)
        otherIndex.append((leftTopPointIndex + 2) % 3)
    else:
        otherIndex.append((leftTopPointIndex + 2) % 3)
        otherIndex.append((leftTopPointIndex + 1) % 3)
    return otherIndex

def rotateAngle(leftTopPoint, rightTopPoint, leftBottomPoint):
    dy = rightTopPoint[1] - leftTopPoint[1]
    dx = rightTopPoint[0] - leftTopPoint[0]
    k = dy / dx
    angle = math.atan(k) * 180 / math.pi
    if leftBottomPoint[1] < leftTopPoint[1]:
        angle -= 180
    return angle

def distance(x, y):
    return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

def trianSquare(a, b, c):
    return a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])

def cross_point(line1, line2):  # 计算交点函数
    x1 = line1[0][0]
    y1 = line1[0][1]
    x2 = line1[1][0]
    y2 = line1[1][1]

    x3 = line2[0][0]
    y3 = line2[0][1]
    x4 = line2[1][0]
    y4 = line2[1][1]

    k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
    b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键
    if (x4 - x3) == 0:  # L2直线斜率不存在操作
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    if k2 == None:
        x = x3
    else:
        x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    return [int(x), int(y)]

def getXY(line, len):
    # 沿着左上角的原点，作目标直线的垂线得到长度和角度
    rho = line[0][0]
    theta = line[0][1]
    # if np.pi / 3 < theta < np.pi * (3 / 4):
    a = np.cos(theta)
    b = np.sin(theta)
    # 得到目标直线上的点
    x0 = a * rho
    y0 = b * rho

    # 延长直线的长度，保证在整幅图像上绘制直线
    x1 = int(x0 + len * (-b))
    y1 = int(y0 + len * (a))
    x2 = int(x0 - len * (-b))
    y2 = int(y0 - len * (a))
    return x1, y1, x2, y2

def findCorners(points):
    cent = []
    temp = []
    for i in points:
        # 求重心
        m = cv2.moments(i)
        cx = np.int0(m['m10'] / m['m00'])
        cy = np.int0(m['m01'] / m['m00'])
        temp.append((cx, cy))
        cent.append([(cx, cy), i])

    lefttop = leftTop(temp)
    other = orderPoints(temp, lefttop)
    cent = {'ul': cent[lefttop],
            'ur': cent[other[0]],
            'dl': cent[other[1]]}

    return cent

def findRotateAngle(points):
    cent = findCorners(points)
    angle = rotateAngle(cent['ul'][0], cent['ur'][0], cent['dl'][0])
    return angle

def getArea(img):
    # 图像灰度化
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 图像二值化
    ret, img2 = cv2.threshold(img_gray, 127, 255, 0)
    # 中值滤波
    img2 = cv2.medianBlur(img2, 3)
    # 寻找轮廓

    contours, hierarchy = cv2.findContours(img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    base = img.copy()
    height, width, _ = img.shape
    base = cv2.resize(base, (width, height))

    hierarchy = hierarchy[0]
    # 轮廓中有两个子轮廓的轮廓可能为二维码位置探测图形
    points = []
    minlen = 100000
    for i in range(0, len(contours)):
        k = i
        c = 0
        while hierarchy[k][2] != -1:
            k = hierarchy[k][2]
            c = c + 1
        if c == 2:
            perimeter = cv2.arcLength(contours[i], True)  # 计算轮廓周长
            if(perimeter<minlen):
                minlen=perimeter
            approx = cv2.approxPolyDP(contours[i], 0.02 * perimeter, True)  # 获取轮廓角点坐标
            if len(approx) == 4:
                points.append(contours[i])
    if len(points)>3:
        rem=[]
        for i in points:
            perimeter = cv2.arcLength(i, True)
            if perimeter>minlen*3:
                rem.append(i)
                continue
        for i in rem:
            points.remove(i)
    for i in points:
        cv2.drawContours(base, [i], -1, (255, 0, 0), 2)
    showImg(base)

    return points


def wrap(img):
    base = cv2.imread('base.png')
    height, width, _ = img.shape
    base = cv2.resize(base, (width, height))
    points = getArea(img)

    base2 = base
    cv2.drawContours(base2, points, -1, (0, 0, 0), 2)
    base2 = cv2.Canny(base2, 10, 100, apertureSize=3)
    lines = cv2.HoughLines(base2, 1, np.pi / 180, 29)

    for line in lines:
        x0, y0, x1, y1 = getXY(line, 2000)
        cv2.line(img, (x0, y0), (x1, y1), (0, 0, 255), 1)
    showImg(img)

def cmp(x, y):
    return (x[0] - y[0]) / [x[1] - y[1]]

def dealLittleRects(rect):
    littleRect = {}
    res = []
    max = 0
    for i in range(0, len(rect)):
        for j in range(i + 1, len(rect)):
            dis = distance(rect[i][0], rect[j][0])
            if (dis > max):
                max = dis
                x1 = rect[i][0]
                x2 = rect[j][0]
    corex = min(x1[0], x2[0]) + math.fabs((x1[0] - x2[0]) / 2)
    corey = min(x1[1], x2[1]) + math.fabs((x1[1] - x2[1]) / 2)
    max = 0
    for i in rect:
        squ = math.fabs(trianSquare(x1, x2, i[0]))
        if squ > max:
            max = squ
            x3 = i[0]
    max = 0
    for i in rect:
        dis = distance(x3, i[0])
        if dis > max:
            max = dis
            x4 = i[0]
    res.append(x1)
    res.append(x2)
    res.append(x3)
    res.append(x4)

    for i in range(0, len(res)):
        if res[i][0] < corex and res[i][1] < corey:
            lefttop = i
    littleRect['ul'] = res[lefttop]

    del res[lefttop]
    res = sorted(res, key=lambda x: x[1] / x[0])
    littleRect['ur'] = res[0]
    littleRect['dr'] = res[1]
    littleRect['dl'] = res[2]
    return littleRect

def perspectTrans(img):
    img2 = img.copy()
    points = getArea(img)
    cent = findCorners(points)

    # 定位三角形
    cv2.line(img2, cent['ul'][0], cent['ur'][0], (0, 255, 0), 3)
    cv2.line(img2, cent['ul'][0], cent['dl'][0], (0, 255, 0), 3)
    cv2.line(img2, cent['dl'][0], cent['ur'][0], (0, 255, 0), 3)
    showImg(img2)

    pnt = dealLittleRects(cent['dl'][1])
    pdl = pnt['dl']
    line1 = (pnt['dl'], pnt['dr'])

    pnt = dealLittleRects(cent['ur'][1])
    pur = pnt['ur']
    line2 = (pnt['ur'], pnt['dr'])

    pdr = cross_point(line1, line2)

    pnt = dealLittleRects(cent['ul'][1])
    pul = pnt['ul']

    # 四个顶点
    cv2.circle(img2, pdl, 5, (255, 0, 0), 5)
    cv2.circle(img2, pdr, 5, (255, 0, 0), 5)
    cv2.circle(img2, pul, 5, (255, 0, 0), 5)
    cv2.circle(img2, pur, 5, (255, 0, 0), 5)
    showImg(img2)

    plane = np.array([[0, 0], [600, 0], [600, 600], [0, 600]], dtype="float32")
    source = np.array([pul, pur, pdr, pdl], dtype="float32")
    M = cv2.getPerspectiveTransform(source, plane)
    # 进行透视变换
    img = cv2.warpPerspective(img, M, (600, 600))
    return img

img = cv2.imread('/home/test_qrcode/image/50large.jpg')
points = getArea(img)
angle = findRotateAngle(points)
height, width, _ = img.shape
rotate_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle=angle, scale=1)
rotated_image = cv2.warpAffine(
                src=img, M=rotate_matrix, 
                dsize=(max(width, height), max(width, height)))
new_img = perspectTrans(rotated_image)
cv2.imwrite('/home/test_qrcode/qrdraw.jpg', new_img)