import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mptch


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


def main(x, y, w, h, theta):
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

    for point in points:
        print('point', point.x, point.y)
    # 用matplotlib绘制出来
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
    # print(x1, y1)
    rect = plt.Rectangle((x1, y1), det_box.w, det_box.h, angle=0, fill=False, color="blue")
    ax.add_patch(rect)
    # 绕 center (x,y) 旋转45度
    polygon = mptch.Polygon(xy=[(points[i].x, points[i].y) for i in range(4)],
                            closed=True, color="red", fill=False)
    ax.add_patch(polygon)

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
    x = -5
    y = -5
    w = 15
    h = 20
    theta = 45
    main(x, y, w, h, theta)

"""
point 7.374368670764582 -3.232233047033631
point -6.767766952966369 -17.374368670764582
point -17.374368670764582 -6.767766952966369
point -3.232233047033631 7.374368670764582
"""
