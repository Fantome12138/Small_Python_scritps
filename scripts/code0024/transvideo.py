import cv2 

# 视频格式转换，例avi --> mp4
cap = cv2.VideoCapture('/home/yolov5-6.0/test1.avi')  #读取视频文件
fps = 16
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
videoWriter = cv2.VideoWriter('video.mp4', fourcc, fps, size)

while(True):
    ret, frame = cap.read()
    if ret:
        videoWriter.write(frame)
        # cv2.imshow("frame", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    else:
        break
cap.release()
videoWriter.release()
print('Done')