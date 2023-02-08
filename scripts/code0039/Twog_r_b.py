'''
一、提取绿色
首先经过BGR分离，做一个2g-r-b的处理，然后进行二值化处理得到最终结果。
缺点：对阳光照射的影响有很大的影响，最好选取阳光充足的照片
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 使用2g-r-b分离土壤与背景

src = cv2.imread('C:\\Users\\zjk\\PycharmProjects\\untitled1\\1.bmp')
cv2.imshow('src', src)

# 转换为浮点数进行计算
fsrc = np.array(src, dtype=np.float32) / 255.0
(b, g, r) = cv2.split(fsrc)
gray = 2 * g - b - r

# 求取最大值和最小值
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

# 计算直方图
hist = cv2.calcHist([gray], [0], None, [256], [minVal, maxVal])
plt.plot(hist)
plt.show()


# 转换为u8类型，进行otsu二值化
gray_u8 = np.array((gray - minVal) / (maxVal - minVal) * 255, dtype=np.uint8)
(thresh, bin_img) = cv2.threshold(gray_u8, -1.0, 255, cv2.THRESH_OTSU)
cv2.imshow('bin_img', bin_img)

# 得到彩色的图像
(b8, g8, r8) = cv2.split(src)
color_img = cv2.merge([b8 & bin_img, g8 & bin_img, r8 & bin_img])
cv2.imshow('color_img', color_img)

cv2.waitKey()
cv2.destroyAllWindows()


'''
二、视频
'''
import cv2
import numpy as np
import time
# 使用2g-r-b分离土壤与背景
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)#设定分辨率，随便给个分辨率，它会自己适应到一个属于自己的比较合适的分辨率
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
while(cap.isOpened()):
    t = time.time()
    ret,frame = cap.read()
    # print(frame.shape)
    fsrc = np.array(frame, dtype=np.float32) / 255.0#这里除以255是为了下面2g - r - b做准备，而且于此同时将数组转换为float小数类型
    print(fsrc)
    (b, g, r) = cv2.split(fsrc)
    gray = 2 * g - b - r#由于一张图片最大255，如果我2*g - b -r 超出了255 默认是255，但是如果是小数，怎么也不会超出255,因为uint8类型超过255就是这个数减255截断
    # cv2.imshow('graw',gray)

    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    print(maxVal-minVal)
    gray_u8 = np.array((gray - minVal) * 255/ (maxVal - minVal) , dtype=np.uint8) #最大的*255可能会超出255，所以要除以一个max-min，保证最大是255
    # cv2.imshow('gray_u8',gray_u8)
    (thresh, bin_img) = cv2.threshold(gray_u8, -1.0, 255, cv2.THRESH_OTSU)# cv2.THRESH_OTSU是自动分割阈值
    cv2.imshow('bin_img', bin_img)

    # 得到彩色的图像
    (b8, g8, r8) = cv2.split(frame)
    color_img = cv2.merge([b8 & bin_img, g8 & bin_img, r8 & bin_img])#将原来彩色图片与分割出来的进行合并
    cv2.imshow('color_img', color_img)
    f = time.time()
    print("时间是",f-t)
    c = cv2.waitKey(1)
    if c == 27:
        break
cap.release()
cv2.destroyAllWindows()


'''
三、HSV+ 2G-R-B + 开运算 + 膨胀
'''
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)#设定分辨率，随便给个分辨率，它会自己适应到一个属于自己的比较合适的分辨率
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while(cap.isOpened()):
    t = time.time()
    ret, img = cap.read()
    img = cv2.blur(img, (5, 5))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    minGreen = np.array([40, 50, 50])
    maxGreen = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, minGreen, maxGreen)
    k = np.ones((4, 4), np.uint8)
    mask1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    k2 = np.ones((10, 10), np.uint8)
    mask2 = cv2.dilate(mask1, k2)
    green = cv2.bitwise_and(img, img, mask=mask2)
    cv2.imshow('img', img)
    cv2.imshow('green', green)
    fsrc = np.array(green, dtype=np.float32) / 255.0
    (b, g, r) = cv2.split(fsrc)
    gray = 2 * g - b - r
    cv2.imshow('gray', gray)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    gray_u8 = np.array((gray - minVal) / (maxVal - minVal) * 255, dtype=np.uint8)
    (thresh, bin_img) = cv2.threshold(gray_u8, -1.0, 255, cv2.THRESH_OTSU)
    cv2.imshow('bin_img', bin_img)
    (b8, g8, r8) = cv2.split(green)
    color_img = cv2.merge([b8 & bin_img, g8 & bin_img, r8 & bin_img])
    cv2.imshow('color_img', color_img)
    print('消耗时间是：', time.time()-t)
    k = cv2.waitKey(1)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()

'''
四、计算面积，并且挑出需要的轮廓
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 使用2g-r-b分离土壤与背景

src = cv2.imread('C:\\Users\\zjk\\PycharmProjects\\zjk\\cao1.jpg')
src = cv2.resize(src,(1280,1280))
cv2.imshow('src', src)

# 转换为浮点数进行计算
fsrc = np.array(src, dtype=np.float32) / 255.0
(b, g, r) = cv2.split(fsrc)
gray = 2 * g - b - r

# 求取最大值和最小值
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

# 计算直方图
hist = cv2.calcHist([gray], [0], None, [256], [minVal, maxVal])
plt.plot(hist)
#plt.show()


# 转换为u8类型，进行otsu二值化
gray_u8 = np.array((gray - minVal) / (maxVal - minVal) * 255, dtype=np.uint8)
(thresh, bin_img) = cv2.threshold(gray_u8, -1.0, 255, cv2.THRESH_OTSU)
cv2.imshow('bin_img', bin_img)
print(np.sum(bin_img==255))

# 得到彩色的图像
(b8, g8, r8) = cv2.split(src)
color_img = cv2.merge([b8 & bin_img, g8 & bin_img, r8 & bin_img])
cv2.imshow('color_img', color_img)

bin_img2 = bin_img.copy()
contours,hierarchy = cv2.findContours(bin_img2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
src2 = src.copy()

t = len(contours)
contoursImg = []

final = np.zeros(bin_img.shape,np.uint8)#新建一个全黑画布
for i in range(t):
    temp = np.zeros(bin_img.shape,np.uint8)#创建一块黑布
    contoursImg.append(temp)#每一次都把创建的黑布放在contoursImg中
    contoursImg[i] = cv2.drawContours(contoursImg[i],contours,i,(255,255,255),-1)
    area = cv2.contourArea(contours[i])
    if area > 10000:#判断面积大小，如果符合要求，将final黑画布与contoursImg[i]相加，由于是uint8，最大不会超过255
        final = cv2.bitwise_or(final,contoursImg[i])
        #cv2.imshow('contours+['+str(i)+']',contoursImg[i])
final_1 = cv2.bitwise_and(final,bin_img)
#final_1[1,1] = 255
cv2.imshow('final',final_1)
print(np.sum(final_1==255))
print(np.sum(bin_img==255)-np.sum(final_1==255))
cv2.waitKey()
cv2.destroyAllWindows()

# 335670代表bin_img中的元素255（即白点的个数）的个数
# 304915代表bin_img中轮廓面积超过10000的面积（也是指的白点的个数）
# 30755代表上边两者的差值


'''
五、视频读取并且保存处理后的图像
'''
import cv2
import numpy as np

def green(img):
    # img = cv2.blur(img,(10,10))
    # cv2.imshow('img', img)
    fsrc = np.array(img, dtype=np.float32) / 255.0
    (b, g, r) = cv2.split(fsrc)
    gray = 2 * g - b - r
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    gray_u8 = np.array((gray - minVal) * 255 / (maxVal - minVal), dtype=np.uint8)
    (thresh, bin_img) = cv2.threshold(gray_u8, -1.0, 255, cv2.THRESH_OTSU)
    (b8, g8, r8) = cv2.split(img)
    color_img = cv2.merge([b8 & bin_img, g8 & bin_img, r8 & bin_img])
    # cv2.imshow("color_img", color_img)
    return color_img

cap = cv2.VideoCapture('output2.avi')

fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 初始化
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # 得到摄像头的高度和宽度
out = cv2.VideoWriter('output22.avi', fourcc, 20, size)  # 带入初始化，设置fps，带入摄像头高度
while (cap.isOpened()):
    ret, img = cap.read()
    if ret == True:

        img = green(img)
        out.write(img)
        k = cv2.waitKey(1)
        if k == 27:
            break
    else:
        break

out.release()
cap.release()
cv2.destroyAllWindows()









