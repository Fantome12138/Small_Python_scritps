import cv2
import numpy as np

''''''

imgi = cv2.imread('C:\\Users\\test_img\\aai.jpg', 1)
imgv = cv2.imread('C:\\Users\\test_img\\plotaav.jpg', 1)
# 鼠标事件
def on_EVENT_LBUTTONDOWN1(event, x, y, flags, param):
    point_queue1 = queue.Queue()
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        xy = "%d,%d" % (x, y)
        # 画圈（图像:img，坐标位置:xy，半径:1(就是一个点)，颜色:R，厚度：-1(就是实心)
        cv2.circle(imgv, (x, y), 1, (0, 0, 255), thickness=-1)
        cv2.putText(imgv, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image_v", imgv)
        # 写入queue
        x_point = float(x)
        y_point = float(y)
        points = [x_point, y_point]
        print('points:', x_point, y_point)
        f = open("./coordinate1.txt", "a+")
        f.writelines(str(points) + '\n')
        point_queue1.put(points)

def on_EVENT_LBUTTONDOWN2(event, x, y, flags, param):
    point_queue2 = queue.Queue()
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击,读取坐标并打点
        xy = "%d,%d" % (x, y)
        cv2.circle(imgi, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(imgi, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image_i", imgi)
        x_point = float(x)
        y_point = float(y)
        points = [x_point, y_point]
        print('points:', x_point, y_point)
        f = open("./coordinate2.txt", "a+")
        f.writelines(str(points) + '\n')
        point_queue2.put(points)

cv2.namedWindow("image_v", 0)
cv2.namedWindow("image_i", 0)
cv2.imshow("image_v", imgv)
cv2.imshow("image_i", imgi)
cv2.setMouseCallback("image_v", on_EVENT_LBUTTONDOWN1)
cv2.setMouseCallback("image_i", on_EVENT_LBUTTONDOWN2)

cv2.waitKey(0)
cv2.destroyAllWindows()
