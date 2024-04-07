import numpy as np

def nearest_neighbor_interpolation(img, new_size):
    old_size = img.shape
    row_ratio, col_ratio = old_size[0]/new_size[0], old_size[1]/new_size[1]

    new_img = np.zeros(new_size)
    for i in range(new_size[0]):
        for j in range(new_size[1]):
            new_img[i, j] = img[int(i*row_ratio), int(j*col_ratio)]
    return new_img

def bilinear_interpolation(img, new_size):
    old_size = img.shape
    row_ratio, col_ratio = old_size[0]/new_size[0], old_size[1]/new_size[1]

    new_img = np.zeros(new_size)
    for i in range(new_size[0]):
        for j in range(new_size[1]):
            x_l, y_l = int(i*row_ratio), int(j*col_ratio)
            x_h, y_h = min(x_l+1, old_size[0]-1), min(y_l+1, old_size[1]-1)

            x_weight = i*row_ratio - x_l
            y_weight = j*col_ratio - y_l

            a = img[x_l, y_l]
            b = img[x_l, y_h]
            c = img[x_h, y_l]
            d = img[x_h, y_h]

            pixel = a*(1-x_weight)*(1-y_weight) + b*(1-x_weight)*y_weight + \
                    c*x_weight*(1-y_weight) + d*x_weight*y_weight
            new_img[i, j] = pixel
    return new_img
