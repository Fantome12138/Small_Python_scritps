import cv2 as cv
import scipy.ndimage

'''火灾视频为例，得到前后两帧变化部分区域，进而可以进行运动(移动)物体检测。'''

def medium_filter(im, x, y, step):
    sum_s = []
    for k in range(-int(step / 2), int(step / 2) + 1):
        for m in range(-int(step / 2), int(step / 2) + 1):
            sum_s.append(im[x + k][y + m])
    sum_s.sort()
    return sum_s[(int(step * step / 2) + 1)]

def frame_diff(img_1, img_2, Thre):
    gray1 = cv.cvtColor(img_1, cv.COLOR_BGR2GRAY)  # 灰度化
    gray1 = cv.GaussianBlur(gray1, (3, 3), 0)  # 高斯滤波
    gray2 = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)
    gray2 = cv.GaussianBlur(gray2, (3, 3), 0)
    d_frame = cv.absdiff(gray1, gray2)
    ret, d_frame = cv.threshold(d_frame, Thre, 255, cv.THRESH_BINARY)
    result = scipy.ndimage.median_filter(d_frame, (7, 7))    # 对结果进行中值滤波

    return result


if __name__ == '__main__':
    capture = cv.VideoCapture("house5.mp4")
    Thre = 8 #Thre表示像素阈值
    ret1, frame1 = capture.read()
    frame2_copy = frame1

    while(True):
        frame1 = frame2_copy
        ret2, frame2 = capture.read()

        if not ret2:
            print("... end of video file reached")
            break

        d_frame = frame_diff(frame1, frame2, Thre)

        cv.namedWindow('Result of original img', 0)
        cv.imshow('Result of original img', frame2)
        cv.waitKey(2)
        cv.namedWindow('Result of frame diff', 0)
        cv.imshow('Result of frame diff', d_frame)
        cv.waitKey(2)

        frame2_copy = frame2.copy()
