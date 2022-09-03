#!/usr/bin/python3
# -*- coding: utf-8 -*-
import cv2
from test2 import func1
from pylogging import loggings

mylogging = loggings('main').getlog()

def img_brightness(image):
    contrast = 1.5  # 对比度
    brightness = 20  # 亮度
    pic_turn = cv2.addWeighted(image, contrast, image, 0, brightness)
    cc = func1(brightness)
    mylogging.info('done %s', cc)
    return pic_turn

img = cv2.imread('/home/mskj/wangyang/OccludedQR150/plot.jpg')
imgs = img_brightness(img)





