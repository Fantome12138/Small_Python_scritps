'''python多进程实时读取数据'''
from multiprocessing import Queue, Process
import cv2
import datetime
#################################################
#摄像头地址
url = 'rtsp://admin:123@10.180.12.165'

def producer(q):
    cap = cv2.VideoCapture(url)
    while True:
        print('producer execuation')
        if cap.isOpened():
            ret, img = cap.read()
            q.put(img)


def consumer(q):
    while True:
        print("consumer execuation")
        img = q.get()

        if img is None:
            print("there is no img!")
            break

        width = int(img.shape[1])
        height = int(img.shape[0])
        time_stamp = datetime.datetime.now()
        date_now = time_stamp.strftime('%Y.%m.%d-%H:%M:%S')
        cv2.putText(img, date_now, (int(width / 20), int(height / 8)),cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 10, cv2.LINE_AA)
        img_res = cv2.resize(img, (int(img.shape[1] / 3), int(img.shape[0] / 3)))

        cv2.imshow('img_multi_process', img_res)
        cv2.waitKey(1)

if __name__ == "__main__":
    q = Queue(maxsize=10) #设置对队列最大容量
    p1 = Process(target=producer, args=(q,))
    c1 = Process(target=consumer, args=(q,))
    p1.start()
    c1.start()


'''python多线程实时读取数据'''

import cv2
from threading import Thread
from collections import deque
import datetime
#################################################
# 摄像头地址
url = 'rtsp://admin:123@10.180.12.165'

def producer(cap, q):
    while True:
        print('producer execuation')
        if cap.isOpened():
            ret, img = cap.read()
            q.append(img)

def consumer(q):
    while True:
        if len(q) == 0:
            pass
        else:
            img = q.pop()
            print('consumer execuation')
            width = int(img.shape[1])
            height = int(img.shape[0])
            time_stamp = datetime.datetime.now()
            date_now = time_stamp.strftime('%Y.%m.%d-%H:%M:%S')
            cv2.putText(img, date_now, (int(width / 20), int(height / 8)),cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 10, cv2.LINE_AA)
            img_res = cv2.resize(img, (int(img.shape[1] / 3), int(img.shape[0] / 3)))
            cv2.imshow('img_multi_process', img_res)
            cv2.waitKey(1)


if __name__ == '__main__':

    frame_deque = deque(maxlen=10)
    cap = cv2.VideoCapture(url)
    p1 = Thread(target=producer, args=(cap, frame_deque))
    p2 = Thread(target=consumer, args=(frame_deque,) )
    p1.start()
    p2.start()
    p1.join()
    p2.join()

