import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mptch
from typing import List


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

    def __str__(self):
        return f"Point({self.x}, {self.y})"

    __repr__ = __str__


class Line:

    def __init__(self, start_point:Point, end_point:Point):
        self.start_point:Point = start_point
        self.end_point:Point = end_point

    def get_vector_point(self):
        return Point(self.end_point.x - self.start_point.x, self.end_point.y - self.start_point.y)


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

    points_1 = points + points[0:1]
    lines = [Line(points_1[i], points_1[i+1]) for i in range(4)]

    fig, ax = plt.subplots()
    fig.set_figheight(6)
    fig.set_figwidth(6)
    # 绘制矩形

    # 绕 center (x,y) 旋转45度
    polygon = mptch.Polygon(xy=[(points[i].x, points[i].y) for i in range(4)], closed=True, color="red", fill=False)
    ax.add_patch(polygon)
    np.random.seed(20)
    test_points = [Point(np.random.uniform(-30, 30), np.random.uniform(-30, 30)) for i in range(30)]
    for test_point in test_points:

        if in_polygon(lines, test_point):
            print(f'{test_point} 在多边形中')
            plt.scatter(test_point.x, test_point.y, color="red")
        else:
            print(f'{test_point} 不在多边形中')
            plt.scatter(test_point.x, test_point.y, color="blue")

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
