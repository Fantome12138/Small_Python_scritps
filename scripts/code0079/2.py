import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mptch


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


def main():

    A = Point(0, 0)
    B = Point(2, 0)
    C = Point(2, 2)
    D = Point(0, 2)
    E = Point(1, 3)

    plt.plot([A.x, B.x], [A.y, B.y])
    plt.plot([C.x, D.x], [C.y, D.y])
    plt.plot([A.x, E.x], [A.y, E.y])

    plt.scatter(A.x, A.y)
    plt.scatter(B.x, B.y)
    plt.scatter(C.x, C.y)
    plt.scatter(D.x, D.y)
    plt.scatter(E.x, E.y)

    plt.text(A.x, A.y, 'A')
    plt.text(B.x, B.y, 'B')
    plt.text(C.x, C.y, 'C')
    plt.text(D.x, D.y, 'D')
    plt.text(E.x, E.y, 'E')

    # 判断 AB 是否与 CD相交 如果相交求出交点
    is_intersect, intersection_point = get_intersection_points(Line(A, B), Line(C, D))
    if is_intersect:
        print(f'AB 与 CD 交点为 {intersection_point}')
        plt.scatter(intersection_point.x, intersection_point.y)
        plt.text(intersection_point.x, intersection_point.y, 'ABxCD')
    else:
        print('AB 与 CD 没有相交')

    # 判断 AE 是否与 CD相交 如果相交求出交点
    is_intersect, intersection_point = get_intersection_points(Line(A, E), Line(C, D))
    if is_intersect:
        print(f'AE 与 CD 交点为 {intersection_point}')
        plt.scatter(intersection_point.x, intersection_point.y)
        plt.text(intersection_point.x, intersection_point.y, 'AExCD')
    else:
        print('AE 与 CD 没有相交')

    plt.show()


if __name__ == "__main__":
    main()
