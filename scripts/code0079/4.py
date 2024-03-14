import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mptch
from typing import List
import math


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


def cross2d(A: Point, B: Point):
    return A.x * B.y - B.x * A.y


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
            print(y, x, x_pred, x1, y1, x2, y2)
            if x_pred < x:
                odd_nodes = not odd_nodes

    return odd_nodes


def main():

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

    points_1 = points + points[0:1]
    points_2 = [A, B, C, D] + [A]
    print('points_2', points_2)

    fig, ax = plt.subplots()
    fig.set_figheight(6)
    fig.set_figwidth(6)

    # 绘制矩形
    # 第一个参数为左下角坐标 第二个参数为width 第三个参数为height
    x_center, y_center, width, height = det_box.x, det_box.y, det_box.w, det_box.h
    x1 = x_center - det_box.w / 2
    y1 = y_center - det_box.h / 2
    x2 = x_center + det_box.w / 2
    y2 = y_center + det_box.h / 2

    rect = plt.Rectangle((x1, y1), det_box.w, det_box.h, angle=0, fill=False, color="blue")
    ax.add_patch(rect)

    # 绕 center (x,y) 旋转45度
    polygon = mptch.Polygon(xy=[(points[i].x, points[i].y) for i in range(4)], closed=True, color="red", fill=False)
    ax.add_patch(polygon)

    insection_points = []

    for i in range(4):
        for j in range(4):
            line1 = points_1[i:i+2]
            line2 = points_2[j:j+2]

            is_intersect, intersection_point = get_intersection_points(Line(line1[0], line1[1]), Line(line2[0], line2[1]))

            if is_intersect:
                plt.scatter(intersection_point.x, intersection_point.y)
                insection_points.append(intersection_point)

    lines_1 = [Line(points_1[i], points_1[i+1]) for i in range(4)]
    lines_2 = [Line(points_2[i], points_2[i+1]) for i in range(4)]

    for i in range(4):
        point = points_1[i]
        if in_polygon(lines_2, point):
            print(f'{point} 在多边形中')
            insection_points.append(point)
        else:
            print(f'{point} 不在多边形中')

    for i in range(4):
        point = points_2[i]
        if in_polygon(lines_1, point):
            print(f'{point} 在多边形中')
            insection_points.append(point)
        else:
            print(f'{point} 不在多边形中')


    center_point = Point(0, 0)
    for point in insection_points:
        center_point += point
    center_point /= len(insection_points)

    vectors = [point - center_point for point in insection_points]
    vectors_degree = [math.atan2(vector.y, vector.x) for vector in vectors]
    vectors = list(zip(vectors, insection_points, vectors_degree))
    vectors.sort(key=lambda x: x[-1])

    for i, (_, point, _) in enumerate(vectors):
        plt.scatter(point.x, point.y, color="red")
        plt.text(point.x, point.y, f'{i}')

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
