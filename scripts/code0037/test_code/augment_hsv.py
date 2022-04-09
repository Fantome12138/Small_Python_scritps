import cv2
import random
import numpy as np

def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    """用在LoadImagesAndLabels模块的__getitem__函数
    hsv色域增强  处理图像hsv，不对label进行任何处理
    :param img: 待处理图片  BGR [736, 736]
    :param hgain: h通道色域参数 用于生成新的h通道
    :param sgain: h通道色域参数 用于生成新的s通道
    :param vgain: h通道色域参数 用于生成新的v通道
    :return: 返回hsv增强后的图片 img
    """
    if hgain or sgain or vgain:
        # 随机取-1到1三个实数，乘以hyp中的hsv三通道的系数  用于生成新的hsv通道
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))  # 图像的通道拆分 h s v
        dtype = img.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)         # 生成新的h通道
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)  # 生成新的s通道
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)  # 生成新的v通道

        # 图像的通道合并 img_hsv=h+s+v  随机调整hsv之后重新组合hsv通道
        # cv2.LUT(hue, lut_hue)   通道色域变换 输入变换前通道hue 和变换后通道lut_hue
        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        # no return needed  dst:输出图像
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed  hsv->bgr
        return img_hsv

path = '/home/test/Some_Python_Scripts/scripts/code0037/test_code/01.jpg'
image = cv2.imread(path)
img_hsv = augment_hsv(image, hgain=0.5, sgain=0.5, vgain=0.5)

cv2.imwrite('/home/test/Some_Python_Scripts/scripts/code0037/test_code/01_save.jpg', img_hsv)
