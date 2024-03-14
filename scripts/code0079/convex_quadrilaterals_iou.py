import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mptch
from typing import List
import math
from copy import deepcopy
import cv2


class RotatedBox:

    def __init__(self, x, y, w, h, theta):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.theta = theta


class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other: 'Point'):
        return Point(other.x + self.x, other.y + self.y)

    def __sub__(self, other: 'Point'):
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, other_scaler):
        return Point(self.x * other_scaler, self.y * other_scaler)

    def __truediv__(self, other_scaler):
        return Point(self.x / other_scaler, self.y / other_scaler)

    def __str__(self):
        return f"Point({self.x}, {self.y})"

    __repr__ = __str__


class Line:

    def __init__(self, start_point, end_point):
        self.start_point = start_point
        self.end_point = end_point

    def get_vector_point(self):
        return Point(self.end_point.x - self.start_point.x, self.end_point.y - self.start_point.y)

    def __str__(self):
        return f"start_point: Point({self.start_point.x}, {self.start_point.y}), end_point:Point({self.end_point.x}, {self.end_point.y})"


def cross2d(A: Point, B: Point):
    return A.x * B.y - B.x * A.y

def dot2d(A: Point, B: Point):
    return A.x * B.x + A.y * B.y

def get_intersection_points(line1: Line, line2: Line):

    line1_vec = line1.get_vector_point()
    line2_vec = line2.get_vector_point()

    det_value = cross2d(line2_vec, line1_vec)

    if abs(det_value) < 1e-14:
        return False, Point(0, 0)

    ac_line_vec = line2.start_point - line1.start_point

    t = cross2d(line2_vec, ac_line_vec) / det_value
    u = cross2d(line1_vec, ac_line_vec) / det_value

    eps = 1e-14

    if -eps <= t <= 1.0 + eps and -eps <= u <= 1.0 + eps:
        return True, line1.start_point + line1_vec * t

    return False, Point(0, 0)


def in_polygon(lines: List[Line], point: Point):

    odd_nodes = False

    x = point.x
    y = point.y

    for line in lines:
        y1 = line.start_point.y
        y2 = line.end_point.y
        x1 = line.start_point.x
        x2 = line.end_point.x
        if ((y1 < y <= y2) or (y2 < y <= y1)) and min(x1, x2) <= x:
            x_pred = (y - y1) / (y2 - y1) * (x2 - x1) + x1
            if x_pred < x:
                odd_nodes = not odd_nodes

    return odd_nodes


def polygon_area(points):
    copy_points = points + [points[0]]
    lines = [Line(copy_points[i], copy_points[i + 1]) for i in range(len(copy_points) - 1)]

    s_polygon = 0.0
    for line in lines:
        A, B = line.start_point, line.end_point
        s_tri = cross2d(A, B)
        s_polygon += s_tri
    return abs(s_polygon / 2)


def convex_hull_graham(points: List[Point]):

    ret_points = []

    copy_points = deepcopy(points)
    # 1. 先找出y值最小的点，如果存在y值相等，则优先选择x值最小的作为起始点$P_0$，该点一定处于凸包上
    copy_points.sort(key=lambda point:(point.y, point.x))

    start_point = copy_points[0]

    # 2. 以$P_0$作为原点，其他所有点减去$P_0$得到对应的向量
    vectors = []
    len_copy_point = len(copy_points)
    for i in range(0, len_copy_point, 1):
        vector = copy_points[i] - start_point
        vectors.append(vector)

    # 3. 计算所有向量与$X$轴正向的夹角$\alpha$，按从小到大进行排列，
    # 遇到$\alpha$相同的情况，则向量较短（即离$P_0$较近的点）的排在前面，
    # 得到初始点序$P_1,P_2, ..., P_n$，
    # 由几何关系可知点序中第一个点$P_1$和最后一个点$P_n$一定在凸包上；
    def cmp_function(point_a: Point):
        # 先根据方位角，方位角相同再根据模长
        return math.atan2(point_a.y, point_a.x), dot2d(point_a, point_a)

    vectors.sort(key=cmp_function)
    dists = [dot2d(vector, vector) for vector in vectors]

    # 4. 将$P_0$和$P_1$压入栈中，将后续点$P_2$作为当前点，跳转第8步。
    ret_points.append(vectors[0])
    k = 1
    while k < len_copy_point:
        if dists[k] > 1e-8:
            break
        k += 1

    if k >= len_copy_point:
        return ret_points

    ret_points.append(vectors[k])

    m = len(ret_points)

    # 5.栈中最上面两个元素形成向量$P_{ij}, i < j$
    # 利用叉乘判断当前点是否在该向量的左边还是右边或者向量上
    # 6. 如果在左边或者向量上，则将当前点压入栈中，下一个点作为当前点，跳转第8步
    # 7. 如果当前点在向量右边，则表明栈顶元素不在凸包上，将栈顶元素弹出，跳转第5步
    # 8. 判断当前点是否是最后一个元素，如果是则将其压缩栈中，栈中所有元素即是凸包上所有点，算法结束，否则跳到第5步。
    for i in range(k + 1, len_copy_point, 1):
        while m >= 2:
            # 查看当前点在向量左边还是右边
            q1 = vectors[i] - ret_points[-2]
            q2 = ret_points[-1] - ret_points[-2]
            if q1.x * q2.y >= q2.x * q1.y:
                m -= 1
                ret_points.pop()
            else:
                break

        ret_points.append(vectors[i])
        m = len(ret_points)

    for point in ret_points:
        point.x += start_point.x
        point.y += start_point.y

    return ret_points


def iou_polygon(points_1: List[Point], points_2: List[Point]):

    points_1 = convex_hull_graham(points_1)
    points_2 = convex_hull_graham(points_2)

    insection_points = []

    len_points_1 = len(points_1)
    len_points_2 = len(points_2)

    points_1 = points_1 + [points_1[0]]
    points_2 = points_2 + [points_2[0]]

    for i in range(len_points_1):
        for j in range(len_points_2):
            line1 = points_1[i:i+2]
            line2 = points_2[j:j+2]

            is_intersect, intersection_point = get_intersection_points(Line(line1[0], line1[1]), Line(line2[0], line2[1]))

            if is_intersect:
                insection_points.append(intersection_point)

    lines_1 = [Line(points_1[i], points_1[i+1]) for i in range(len_points_1)]
    lines_2 = [Line(points_2[i], points_2[i+1]) for i in range(len_points_2)]

    for i in range(len_points_1):
        point = points_1[i]
        if in_polygon(lines_2, point):
            insection_points.append(point)

    for i in range(len_points_2):
        point = points_2[i]
        if in_polygon(lines_1, point):
            insection_points.append(point)

    insection_points = convex_hull_graham(insection_points)

    if len(insection_points) <= 2:
        return 0

    insection_area = polygon_area(insection_points)

    points_1_area = polygon_area(points_1)
    points_2_area = polygon_area(points_2)

    return insection_area / (points_1_area + points_2_area - insection_area)


def iou_polygon_cv2(contour1, contour2):
    # opencv 版本为 4.5

    # 计算交集面积
    contour1 = np.array([(point.x, point.y) for point in contour1]).astype(np.float32)
    contour2 = np.array([(point.x, point.y) for point in contour2]).astype(np.float32)
    _, intersection = cv2.intersectConvexConvex(contour1, contour2)
    intersection_area = cv2.contourArea(intersection)

    # 计算并集面积
    union_area = cv2.contourArea(contour1) + cv2.contourArea(contour2) - intersection_area

    # 计算IOU
    iou = intersection_area / union_area
    return iou

def main():
    # return 
    x = -5
    y = -5
    w = 15
    h = 20
    theta = 45
    theta = theta / 180 * math.pi
    det_box = RotatedBox(x, y, w, h, theta)

    points = [Point(0, 0) for _ in range(4)]
    cos_theta = math.cos(theta) * 0.5
    sin_theta = math.sin(theta) * 0.5

    points[0].x = det_box.x + sin_theta * det_box.h + cos_theta * det_box.w
    points[0].y = det_box.y + cos_theta * det_box.h - sin_theta * det_box.w

    points[1].x = det_box.x - sin_theta * det_box.h + cos_theta * det_box.w
    points[1].y = det_box.y - cos_theta * det_box.h - sin_theta * det_box.w

    points[2].x = 2 * det_box.x - points[0].x
    points[2].y = 2 * det_box.y - points[0].y

    points[3].x = 2 * det_box.x - points[1].x
    points[3].y = 2 * det_box.y - points[1].y

    # # points 整体向下移动5
    for point in points:
        point.y -= 5

    x1 = x - w / 2
    y1 = y - h / 2

    x2 = x + w / 2
    y2 = y - h / 2

    x3 = x + w / 2
    y3 = y + h / 2

    x4 = x - w / 2
    y4 = y + h / 2

    # 判断 各个直线之间是否有交点
    A = Point(x1, y1)
    B = Point(x2, y2)
    C = Point(x3, y3)
    D = Point(x4, y4)


    fig, ax = plt.subplots()
    fig.set_figheight(6)
    fig.set_figwidth(6)

    # 绘制矩形
    # 第一个参数为左下角坐标 第二个参数为width 第三个参数为height
    x_center, y_center, width, height = det_box.x, det_box.y, det_box.w, det_box.h
    x1 = x_center - det_box.w / 2
    y1 = y_center - det_box.h / 2

    rect = plt.Rectangle((x1, y1), det_box.w, det_box.h, angle=0, fill=False, color="blue")
    ax.add_patch(rect)

    # 绕 center (x,y) 旋转45度
    polygon = mptch.Polygon(xy=[(points[i].x, points[i].y) for i in range(4)], closed=True, color="red", fill=False)
    ax.add_patch(polygon)

    insection_points = []

    points_1 = points
    points_2 = [A, B, C, D]

    # 与opencv结果对比
    iou = iou_polygon(points_1, points_2)
    iou_cv2 = iou_polygon_cv2(points_1, points_2)
    print('iou', iou)
    print('iou_cv2', iou_cv2)



    # 设置坐标轴
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    # 定义坐标
    x = np.arange(-30, 31, 5)
    y = np.arange(-30, 31, 5)
    ax.set_xticks(x)
    ax.set_yticks(y)
    ax.grid(True)
    # 显示图象
    plt.show()


if __name__ == "__main__":
    main()
