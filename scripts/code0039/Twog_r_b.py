import cv2
import numpy as np


def find_green(src):
    ### 将绿色部分mask提取出来，受光照等因素影响小，但受高光物体干扰大
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    hsv = cv2.medianBlur(hsv,5)
    low_hsv = np.array([35,43,46])
    high_hsv = np.array([89,255,255])   # 青绿色在HSV的范围
    
    mask = cv2.inRange(hsv,lowerb=low_hsv,upperb=high_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    closed = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    kernel = np.ones((3,3), dtype=np.uint8)  # 腐蚀
    closed = cv2.erode(closed, kernel, iterations=1)
    
    p = np.where(closed>170)
    p_num = len(p[0])
    if p_num > 2000:
        return 1
    else: return 0

def find_green2(src):
    ### 直接将绿色部分提取出来，但受光照影响大
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    hsv = cv2.medianBlur(hsv,5)
    low_hsv = np.array([35,43,46])
    high_hsv = np.array([89,255,255])   # 青绿色在HSV的范围
    
    mask = cv2.inRange(hsv,lowerb=low_hsv,upperb=high_hsv)
    kernel = np.ones((3,3), dtype=np.uint8)
    closed = cv2.erode(closed, kernel, iterations=1)
    res = cv2.bitwise_and(img, img, mask=mask)
    
    p = np.where(res>100)
    p_num = len(p[0])
    if p_num > 2000:
        return 1
    else: return 0

img = cv2.imread('/Users/fantome/Library/CloudStorage/OneDrive-个人/Git/1_Github/Some_Python_Scripts/scripts/code0039/3cut.jpg')
print(find_green(img))