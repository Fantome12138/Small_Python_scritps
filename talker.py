import sys, os, time
sys.path = [ i for i in sys.path if i.find('2.7')==-1 ]
import pika
import uuid
import cv2
import base64
import random
import json
import yaml
import traceback

config_file = '/home/yolov5_server/server.yaml'
with open(config_file) as f:
    client_params = yaml.load(f, Loader=yaml.FullLoader) # yaml.load(f, Loader=yaml.FullLoader)

robot_send = client_params['client_send2server']


class RpcClient(object):
    def __init__(self):
        self.heartbeat = 0
        self.username = 'username'
        self.password = 'password'
        self.host = client_params['server_info']['ip']
        self.port = client_params['server_info']['port']
        self.credentials = pika.PlainCredentials(self.username, self.password)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(heartbeat=self.heartbeat, 
                          host=self.host, port=self.port, credentials=self.credentials))
        self.channel = self.connection.channel()
        result = self.channel.queue_declare('', exclusive=True, auto_delete=True)
        self.callback_queue = result.method.queue
        self.channel.basic_consume(
            queue=self.callback_queue, 
            on_message_callback=self.on_response, 
            auto_ack=True)
    
    def destory(self,):
        if self.connection.is_closed:
            print('Connection is closed or closing')
        else:
            print('Try to close connection')
            self.connection.close()
            print('Close connection succeed')

    def reinit(self,):
        print("Reload rabbit config")        
        credentials = pika.PlainCredentials(self.username, self.password)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(heartbeat= self.heartbeat, 
                    host = self.host, port = self.port, credentials = credentials))
        self.channel = self.connection.channel()
        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue
        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True)
        print("Reload rabbit succeed")
 
    def on_response(self, channel, method_frame, header_frame, body): # 必须是四个参数
        # 如果收到的ID和本机生成的相同，则返回的结果就是我想要的指令返回的结果
        if self.corr_id == header_frame.correlation_id:
            self.response = body

    # @func_set_timeout(10)
    def wait_server(self, ):
        while self.response is None:  # 若无数据，一直循环
            # 启动后，on_response函数接到消息，self.response 值则不为空
            self.connection.process_data_events()  # 非阻塞版的start_consuming()
            # 收到消息就调用on_response

    def encode_img(self, image, encoding_format='.jpg'):
        img_encode = cv2.imencode(encoding_format, image)[1]
        base64_data = base64.b64encode(img_encode)
        str_encode = base64_data.decode('ascii')
        return str_encode

    def call(self, image):
        if image is not None:
            msg = self.encode_img(image)
            img_name = 'object_detect'
            imgdata = [{'name':img_name,'payload':msg}]
            robot_send.update({'imgdata':imgdata})
            send_data = json.dumps(robot_send, sort_keys=True, indent=4, separators=(',', ':'))
        else:
            robot_send.update({'Error_code':'555'})
            send_data = json.dumps(robot_send, sort_keys=True, indent=4, separators=(',', ':'))
            print('Image is None, Send error code 555')
   
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(
            exchange='',
            routing_key=client_params['server_info']['queue'],  # 发消息到rpc_queue
            properties=pika.BasicProperties(          # 消息持久化
                        reply_to=self.callback_queue, # 让服务端命令结果返回到callback_queue
                        correlation_id=self.corr_id,  # 把随机uuid同时发给服务器
                        ),
            body=send_data)
        try:
            self.wait_server()   
        except:
            print('Wait server time out')
        return self.response  # 返回的是byte


if __name__ == '__main__':
    rpc = RpcClient()    
    path = client_params['test_config']['read_img_path']
    names = client_params['model_detection_type']
    img = cv2.imread(path)
    img_shape = img.shape
    response = rpc.call(img)   
    
    
    
    