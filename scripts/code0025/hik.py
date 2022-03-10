import cv2 
source = "rtsp://admin:123456@192.168.x.x/Streaming/Channels/1"  # 输入IP、用户名、密码
cap = cv2.VideoCapture(source)
while cap.isOpened():
     back, frame = cap.read()  # 返回的参数一bool类型，参数二表示帧数
     cv2.imshow(" windows ",frame)
     if not back:
         break
     if cv2.waitKey(1) & 0xFF==1:
         break
