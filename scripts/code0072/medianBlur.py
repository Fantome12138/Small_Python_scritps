# -*- coding: utf-8 -*-
import numpy as np
import cv2


def mean_Blur(img, K_size=3):
    h, w, c = img.shape
    # 零填充
    pad = K_size // 2
    out = np.zeros((h+2*pad, w+2*pad, c), dtype=np.float)
    out[pad:pad+h, pad:pad+w] = img.copy().astype(np.float)
    # 卷积的过程
    tmp = out.copy()
    for y in range(h):
        for x in range(w):
            for ci in range(c):
                out[pad+y,pad+x,ci] = np.mean(tmp[y:y+K_size, x:x+K_size, ci])
    out = out[pad:pad+h, pad:pad+w].astype(np.uint8)

    return out

def median_Blur(img, filiter_size=3):  #当输入的图像为彩色图像
    image_copy = np.array(img, copy = True).astype(np.float32)
    processed = np.zeros_like(image_copy)
    middle = int(filiter_size / 2)
    r = np.zeros(filiter_size * filiter_size)
    g = np.zeros(filiter_size * filiter_size)
    b = np.zeros(filiter_size * filiter_size)

    for i in range(middle, image_copy.shape[0] - middle):
        for j in range(middle, image_copy.shape[1] - middle):
            count = 0
            #依次取出模板中对应的像素值
            for m in range(i - middle, i + middle +1):
                for n in range(j - middle, j + middle + 1):
                    r[count] = image_copy[m][n][0]
                    g[count] = image_copy[m][n][1]
                    b[count] = image_copy[m][n][2]
                    count += 1
            r.sort()
            g.sort()
            b.sort()
            processed[i][j][0] = r[int(filiter_size*filiter_size/2)]
            processed[i][j][1] = g[int(filiter_size*filiter_size/2)]
            processed[i][j][2] = b[int(filiter_size*filiter_size/2)]
    processed = np.clip(processed, 0, 255).astype(np.uint8)
    return processed

import numpy as np
import scipy.signal as signal

def gaussian_filter(size, sigma):
    """
    Create a 2D Gaussian filter.
    np.mgrid函数可以生成从指定的起始点到结束点的均匀间隔的整数序列，
    对于每个维度，它返回一个一维数组。当这些一维数组沿着最后一个轴合并时，就形成了一个多维网格。
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

def apply_gaussian_filter(image, size, sigma):
    """
    Apply a 2D Gaussian filter to an image.
    """
    # Convolve the image with the Gaussian filter
    return signal.convolve2d(image, gaussian_filter(size, sigma), mode='same')

image = np.random.rand(100, 100)
filtered_image = apply_gaussian_filter(image, 5, 1.0)
print("Filtered image:\n", filtered_image)


def median_Blur_gray(img, filiter_size=3):  #当输入的图像为灰度图像
    image_copy = np.array(img, copy = True).astype(np.float32)
    processed = np.zeros_like(image_copy)
    middle = int(filiter_size / 2)

    for i in range(middle, image_copy.shape[0] - middle):
        for j in range(middle, image_copy.shape[1] - middle):
            temp = []
            for m in range(i - middle, i + middle +1):
                for n in range(j - middle, j + middle + 1):
                    if m-middle < 0 or m+middle+1 >image_copy.shape[0] \
                        or n-middle < 0 or n+middle+1 > image_copy.shape[1]:
                        temp.append(0)
                    else:
                        temp.append(image_copy[m][n])
            temp.sort()
            processed[i][j] = temp[(int(filiter_size*filiter_size/2)+1)]
    processed = np.clip(processed, 0, 255).astype(np.uint8)
    return processed
