# 查看rosbag数据包信息，其中包括topics名称等信息
# rosbag info xxx.bag


# 解析.bag文件得到带时间戳的.jpg格式图片

#coding:utf-8
import roslib
import rosbag
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError

path='/home/song/Autoware/ros/pic/' #存放图片的位置

class ImageCreator():
    def __init__(self):
        self.bridge = CvBridge()
        with rosbag.Bag('test.bag', 'r') as bag:   #要读取的bag文件；
            for topic,msg,t in bag.read_messages():
                if topic == "/image_raw":  #图像的topic；
                        try:
                            cv_image = self.bridge.imgmsg_to_cv2(msg,"bgr8")
                        except CvBridgeError as e:
                            print (e)
                        timestr = "%.6f" %  msg.header.stamp.to_sec()
                        #%.6f表示小数点后带有6位，可根据精确度需要修改；
                        image_name = timestr+ ".jpg" #图像命名：时间戳.jpg
                        cv2.imwrite(path+image_name, cv_image)  #保存；

if __name__ == '__main__':
    try:
        image_creator = ImageCreator()
    except rospy.ROSInterruptException:
        pass


# 提取数据为csv或txt格式， 命令行提取
# rostopic echo -b xxx.bag -p /topic > xxx.csv    # (或.txt)都可以