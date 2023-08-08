import cv2
import json
import time
import base64


def encode_img(image, encoding_format='.jpg'):
    img_encode = cv2.imencode(encoding_format, image)[1]
    base64_data = base64.b64encode(img_encode)
    str_encode = base64_data.decode('ascii')
    return str_encode

def sort_msg_pub(img):
        detect_result_img = encode_img(img)
        _time=time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
        img_name = "helmet_warning{}.jpg".format(_time)
        vision_realtime_msg = {
                                "type": "HELMET_RECOGNITION",
                                "file_name": str(img_name),
                                "staff_number":"wangyang",
                                "encoding": "base64",
                                "content": " ",
                                "is_end": True
                            }
        vision_realtime_msg.update({"content":str(detect_result_img)})
        vision_realtime_msg_js = json.dumps(vision_realtime_msg)
        
        file_name = '/home/robot/vision_logs/helmet_logs/save_alarm_img/save_json{}.json'.format(_time)
        f2 = open(file_name, 'w')
        f2.write(vision_realtime_msg_js)
        f2.close()
        return vision_realtime_msg_js


alarm_msg = sort_msg_pub(img=_visible_img)  # _visible_img是结果图像
ros_pub.publish(alarm_msg)

##  其中 ros_pub 定义为
ros_pub = rospy.Publisher(pub_msg_name, String, queue_size=10)