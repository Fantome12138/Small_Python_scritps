# -*- coding: utf-8 -*-
#!/usr/bin/python3
import sys
sys.path = [ i for i in sys.path if i.find('2.7')==-1 ]
import cv2
import re
import time
import logging
import logging.handlers
import numpy as np
import json
import yaml
import base64
import pika
import traceback
from alkene_detect import YoloDetector

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
# Extra Libraries
sys.path.append(".")
cur_parent_path = '/'.join(sys.path[0].split('/')[:5])
sys.path.append(cur_parent_path)
# Add library
mylogging = logging.getLogger('Detecotion job')
mylogging.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s[line:%(lineno)d] - %(levelname)s:  %(funcName)s - %(message)s")
rotatingHandler = logging.handlers.TimedRotatingFileHandler(filename=cur_parent_path+'/log/Server_log.log', when='MIDNIGHT',
                                       interval=1, backupCount=5)
rotatingHandler.setFormatter(formatter)
mylogging.addHandler(rotatingHandler)

config_file = cur_parent_path+'/server.yaml'
with open(config_file) as f:
    detect_params = yaml.load(f, Loader=yaml.FullLoader)
robot_send = detect_params['server_message_callback']

save_img = detect_params['program_config']['saveimg']
save_img_path = detect_params['program_config']['saveimgpath']


def saveimg(path, images):
    img_name = "{_time}.jpg".format(_time=time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time())))
    path_save = path + str(img_name)
    cv2.imwrite(path_save, images)

def detect(weights_path, img_path):
    '''
    Detection Task
        detection mehtod result format:
        - List: [cls, conf, [int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])]] 
        - xyxy: left up corner and right down corner coordinate  
    '''
    detector = YoloDetector(weights=weights_path, imgs=img_path, conf=detect_params['program_config']['conf'], \
                            iou=detect_params['program_config']['iou'])
    _, _, vision_proxy_res = detector.predict()
    data = []
    x_points = []
    y_points = []
    robot_send_local = []
    robot_send_local = detect_params['server_message_callback']
    names = detect_params['model_detection_type']

    if len(vision_proxy_res) != 0:
        for res_ind in range(len(vision_proxy_res)):
            cls, conf  = vision_proxy_res[res_ind][0], vision_proxy_res[res_ind][1]
            x1, y1, x2, y2 = vision_proxy_res[res_ind][2] 
            cls_tmp = re.findall(r"\d+", str(cls))  # 转换数据的格式
            cls = list(map(int, cls_tmp))
            conf_tmp = re.findall(r"\d+\.?\d*", str(conf))
            conf = list(map(float, conf_tmp))
            cls_return = str(cls)[1]
            cls_return = names[int(cls_return)]  # 返回类别名称，若需返回类别下标记可使用：cls_return = str(cls)[1]
            conf_return = conf_tmp[0]
            x_points.append(x1)  # 顺序不要错: (x1,y1), (x2,y2)
            y_points.append(y1)
            x_points.append(x2)
            y_points.append(y2)
            return_data = {"content":cls_return, "attribute":"None", "conf":conf_return, "x_points":x_points, "y_points":y_points}
            data.append(return_data)
            x_points = []
            y_points = []
        robot_send_local.update({"result":data})
    elif len(vision_proxy_res) == 0:
        robot_send_local.update({"result":[
                  {"content":"None", 
                   "attribute":"None",        
                   "conf":"0",               
                   "x_points":[0,0],   
                   "y_points":[0,0]                      
                  }
              ]})
    return robot_send_local

def on_request(channel, method_frame, header_frame, body):
    try:
        ## 判断是否为client端发送了错误格式的json数据   ## 
        ## 若是，json向client端返回'Error_code':'666'  ## 
        load_data = json.loads(body)
        error_code = int(load_data['Error_code'])
        if error_code == 555:
            mylogging.error('Received bad image')
            raise Exception('Bad image, please need re-send')  
        else:
            imgdata = load_data['imgdata']  # 加载json内'imgdata'标签内信息
            weights_path = detect_params['model_dir']
            imgdata = imgdata[0]
            payload = imgdata['payload']    # 加载imgdata标签内的图像二进制信息
            img_name = imgdata['name']      # 加载imgdata标签内的图像名称
            imgmsg = payload.encode('ascii')
            imgmsg = base64.b64decode(imgmsg)
            img_decode = cv2.imdecode(np.frombuffer(imgmsg, np.uint8), cv2.IMREAD_COLOR)
            img_path = detect_params['program_config']['saveimgpath']
            cv2.imwrite(img_path, img_decode)

            if save_img:
                saveimg(save_img_path, img_decode)
                
            response = detect(weights_path, img_path)
            robot_send.update({"Error_code":"0"})  # 更新错误代号 0(无错误)
    except Exception as err:
        mylogging.info('Bad detection, No detect result, return 666')
        mylogging.error("\n" + traceback.format_exc())
        robot_send.update({"Error_code":"666"})
        response = robot_send

    info = json.dumps(response, sort_keys=True, indent=10, separators=(',', ':'))
    mylogging.info('info %s' % (info))
    channel.basic_publish(exchange='', 
                          routing_key=header_frame.reply_to, 
                          properties = pika.BasicProperties(correlation_id= \
                                                            header_frame.correlation_id), 
                                                            body=str(info))
    channel.basic_ack(delivery_tag=method_frame.delivery_tag)
    mylogging.info('Success detection')


if __name__ == '__main__':
    print("------------Start receiving messages-----------")
    username = detect_params['server_info']['username']
    password = detect_params['server_info']['password']
    host = detect_params['server_info']['ip']
    port = detect_params['server_info']['port']
    credentials = pika.PlainCredentials(username, password)
    connection = pika.BlockingConnection(pika.ConnectionParameters(heartbeat=0, socket_timeout=None ,host=host,port=port, credentials=credentials))
    channel = connection.channel()
    channel.queue_delete(queue=detect_params['server_info']['queue'])
    channel.queue_declare(queue=detect_params['server_info']['queue'])

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=detect_params['server_info']['queue'], on_message_callback=on_request)
    print(" [x] Awaiting RPC requests, To exit press CTRL+C ")
    try:
        channel.start_consuming()
    except pika.exceptions.AMQPConnectionError:
        time.sleep(60)
        channel.start_consuming()
    except KeyboardInterrupt:
        channel.stop_consuming()
        connection.close()
        mylogging.info('Program stoped by manual')
