'''  在调试时，有时需要验证检测位置是否正确，将检测的目标中心画在图片上更直观，因此记录，本例以在图片的中心位置画圆为例。'''

import cv2

img = cv2.imread('book.png')
height = img.shape[0]
width = img.shape[1]
cv2.circle(img, (int(width/2), int(height/2)), 2,(0, 255, 0), 3)
cv2.namedWindow('img',0)
cv2.imshow('img',img)
cv2.waitKey()

# skimage函数的实现
import cv2
from skimage import draw
img = cv2.imread('cat.jpg')
height = img.shape[0]
width = img.shape[1]
rr,cc = draw.circle(int(height/2), int(width/2) ,5)
draw.set_color(img,[rr,cc], [0, 255, 0])
cv2.namedWindow('img',0)
cv2.imshow('img',img)
cv2.waitKey()
