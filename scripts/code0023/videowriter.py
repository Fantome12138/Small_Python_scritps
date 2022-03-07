import os
import cv2 as cv

def image_to_video(file1, output1):
    file = file1
    output = output1
    num = os.listdir(file)  # 生成图片目录下以图片名字为内容的列表
    height = 1080
    weight = 1920
    fps = 20
    fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')  # 用于mp4格式的生成
    videowriter = cv.VideoWriter(output, fourcc, fps, (weight, height))  # 创建一个写入视频对象
    for i in range(len(num)):
        for j, item in enumerate(num):
            path = file + str(item)
            if path:
                print(path)
                frame = cv.imread(path)
                videowriter.write(frame)

    videowriter.release()
    print('Done')


if __name__ == '__main__':
    file = '/home/yolov5-6.0/data/img/'  # 图片目录
    output = '/home/yolov5-6.0/test0001.mp4'  # 生成视频路径
    image_to_video(file, output)