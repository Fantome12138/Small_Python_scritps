#coding:utf-8
import cv2
import datetime
import time


'''
带有微秒级时间戳单个相机连续拍照
'''
cap = cv2.VideoCapture(0)               # 创建一个 VideoCapture 对象
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) # 设置图片的尺寸时应当在一开始就设置否则有时会导致，打开的摄像头默认分辨率较低
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

path = "C:\\Users\\test_img"      
flag = 1        # 设置一个标志，用来输出视频信息1
num = 1         # 递增，用来保存文件名
while(cap.isOpened()):      # 循环读取每一帧s
    ret_flag, frame = cap.read()        
    cv2.imshow("Capture", frame)        
    k = cv2.waitKey(1) & 0xFF       # 每帧数据延时 1ms，延时不能为 0，否则读取的结果会是静态帧
    if k == ord('s'):               # 若检测到按键 ‘s’，打印字符串
        timestr = datetime.datetime.now()              # 获取精确到微秒的系统时间作为时间戳
        filename = path + '\\{0}.jpg'.format(timestr)  # 将时间戳作为照片的文件名
        print(filename)
        cv2.imwrite(filename, frame)
        print(cap.get(3));      # 得到长宽
        print(cap.get(4));
        print("success to save" + str(num) + ".jpg")
        print("-------------------------")
        num += 1
    elif k == ord('q'):     # 若检测到按键 ‘q’，退出
        break
cap.release()               # 释放摄像头
cv2.destroyAllWindows()     # 删除建立的全部窗口


'''
带有微秒级时间戳的双目相机拍照
'''
left_camera = cv2.VideoCapture(0)
left_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
left_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
left_camera.set(cv2.CAP_PROP_FPS, 60)

right_camera = cv2.VideoCapture(1)
right_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
right_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
right_camera.set(cv2.CAP_PROP_FPS, 60)

path="/home/song/pic/" #图片存储路径

AUTO =False          # 自动拍照，或手动按s键拍照
INTERVAL = 0.0000005 # 调整自动拍照间隔
cv2.namedWindow("left")
cv2.namedWindow("right")
cv2.moveWindow("left", 0, 0)

counter = 0
utc = time.time()
folder = "/home/song/pic/" # 拍照文件目录

def shot(pos, frame):
    global counter
    timestr = datetime.datetime.now()
    path = folder + pos + "_" + str(counter) + "_" + str(timestr) + ".jpg"
    cv2.imwrite(path, frame)
    print("snapshot saved into: " + path)

while True:
    ret, left_frame = left_camera.read()
    ret, right_frame = right_camera.read()

    cv2.imshow("left", left_frame)
    cv2.imshow("right", right_frame)

    now = time.time()
    if AUTO and now - utc >= INTERVAL:
        shot("left", left_frame)
        shot("right", right_frame)
        counter += 1
        utc = now

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("s"):
        shot("left", left_frame)
        shot("right", right_frame)
        counter += 1

left_camera.release()
right_camera.release()
cv2.destroyWindow("left")
cv2.destroyWindow("right")
