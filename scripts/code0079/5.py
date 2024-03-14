def polygon_area(points):
    copy_points = points + [points[0]]
    lines = [Line(copy_points[i], copy_points[i + 1]) for i in range(len(copy_points))]

    s_polygon = 0.0
    for line in lines:
        A, B = line.start_point, line.end_point
        s_tri = cross2d(A, B)
        s_polygon += s_tri
    return abs(s_polygon / 2)
