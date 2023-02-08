from math import radians, cos, sin, asin, sqrt

def distance_between_points(lat1, lon1, lat2, lon2):
    """ 
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)  ----WGS-84
    """
    lon1, lat1, lon2, lat2 = map(radians, [float(lon1), float(lat1), float(lon2), float(lat2)])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371.004
    distance = c * r * 1000
    return int(distance)