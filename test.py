'''内外参标定'''
import cv2
import numpy as np
import glob

# 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

# 获取标定板角点的位置
objp = np.zeros((4 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:6, 0:4].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y

obj_points = []  # 存储3D点
img_points = []  # 存储2D点

images = glob.glob("/Users/fantome/Downloads/2/*.jpg")
i = 0
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, (6, 4), None)

    if ret:
        obj_points.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (5,5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
        #print(corners2)
        if [corners2]:
            img_points.append(corners2)
        else:
            img_points.append(corners)

        cv2.drawChessboardCorners(img, (6, 4), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值
        i += 1
        cv2.imwrite('conimg'+str(i)+'.jpg', img)
        # cv2.waitKey(1500)

print(len(img_points))
cv2.destroyAllWindows()

# 标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

print("ret:", ret)
print("mtx:\n", mtx) # 内参数矩阵
print("dist:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
# print("rvecs:\n", rvecs)  # 旋转向量  # 外参数
# print("tvecs:\n", tvecs ) # 平移向量  # 外参数

print("-----------------------------------------------------")

img = cv2.imread(images[2])
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))  # 显示更大范围的图片（正常重映射之后会删掉一部分图像）
print (newcameramtx)
print("------------------使用undistort函数-------------------")
dst = cv2.undistort(img,mtx,dist,None,newcameramtx)
x, y, w, h = roi
dst1 = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult3.jpg', dst1)
print("方法一:dst的大小为:", dst1.shape)

R = cv2.Rodrigues(rvecs)
print(R)









# import cv2
# import numpy as np
# import glob



# # 找棋盘格角点
# # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.0001) # 阈值
# #棋盘格模板规格
# w = 6   # 10 - 1
# h = 4   # 7  - 1
# # 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
# objp = np.zeros((w*h,3), np.float32)
# objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
# objp = objp*38.5  # 18.1 mm

# # 储存棋盘格角点的世界坐标和图像坐标对
# objpoints = [] # 在世界坐标系中的三维点
# imgpoints = [] # 在图像平面的二维点
# #加载pic文件夹下所有的jpg图像
# images = glob.glob("/Users/fantome/Downloads/1/*.jpg")  #   拍摄的十几张棋盘图片所在目录

# i=0
# for fname in images:
#     img = cv2.imread(fname)
#     # 获取画面中心点
#     #获取图像的长宽
#     h1, w1 = img.shape[0], img.shape[1]
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     u, v = img.shape[:2]
#     # 找到棋盘格角点
#     ret, corners = cv2.findChessboardCorners(gray, (w,h),None)
#     # 如果找到足够点对，将其存储起来
#     if ret == True:
#         print("i:", i)
#         i = i+1
#         # 在原角点的基础上寻找亚像素角点
#         cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
#         #追加进入世界三维点和平面二维点中
#         objpoints.append(objp)
#         imgpoints.append(corners)
#         # 将角点在图像上显示
#         # cv2.drawChessboardCorners(img, (w,h), corners, ret)
#         # cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
#         # cv2.resizeWindow('findCorners', 640, 480)
#         # cv2.imshow('findCorners',img)
#         # cv2.waitKey(200)

# #标定
# print('正在计算')
# #标定
# ret, mtx, dist, rvecs, tvecs = \
#     cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# print("ret:",ret  )
# print("mtx:\n",mtx)      # 内参数矩阵
# print("dist畸变值:\n",dist   )   # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
# # print("rvecs旋转（向量）外参:\n",rvecs)   # 旋转向量  # 外参数
# # print("tvecs平移（向量）外参:\n",tvecs  )  # 平移向量  # 外参数
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (u, v), 0, (u, v))
# print('newcameramtx外参',newcameramtx)

