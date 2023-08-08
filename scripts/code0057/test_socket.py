#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from uuid import uuid4
import cv2, os, time, copy, json


def short_uuid():
    uuidChars = ("a", "b", "c", "d", "e", "f",
             "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
             "t", "u", "v", "w", "x", "y", "z", "0", "1", "2", "3", "4", "5",
             "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I",
             "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
             "W", "X", "Y", "Z")
    uuid = str(uuid4()).replace('-', '')
    id = ''
    for i in range(0,8):
        sub = uuid[i * 4: i * 4 + 4]
        x = int(sub,16)
        id += uuidChars[x % 0x3E]
    return id

def save_json_file(data, save_path):
    info = json.dumps(data, sort_keys=True, indent=10, separators=(',', ':'))
    f = open(save_path, 'w')
    f.write(info)

def saveuuid_img(image_save_path='/home/docker/SaveImage/',
                deviceID = 'sonar1##fasdfearfa##/',
                image=None, 
                irimage=None,
                    ):
    data_time = "{_time}".format(_time=time.strftime('%Y_%m_%d', time.localtime(time.time())))
    path = image_save_path+deviceID+data_time+'/'
    if not os.path.exists(path):
        os.makedirs(path)
        # mylogging.info('make new save image dir %s',path)
        print('make new save image dir %s',path)
    else: pass
    u_path = str(short_uuid())
    p = (path+u_path+'/')
    if not os.path.exists(p):
        os.makedirs(p)
        # mylogging.info('make new correlationID dir %s',p)
        print('make new correlationID dir %s',p)
    else: pass
    imgsave = p+'image.jpg'
    irimgsave = p+'irimage.jpg'
    cv2.imwrite(imgsave,image)
    cv2.imwrite(irimgsave,irimage)
    return u_path, data_time+'/'+u_path+'/', data_time

def getWarningImg(self, infoAI={}, deviceID=''):

    image1 = cv2.imread('/home/robot/vision_logs/main_log/tmp/bad.jpg')
    image2 = image1
    save_path = '/home/docker/SaveImage/'
    deviceID = 'laser1##fasdfearfa##/'
    correlationID, save_img_path, data_time = saveuuid_img(save_path, deviceID, image1, image2)
    print('correlationID: ', correlationID)

    concentration = 50
    p = 0
    t = 0
    z = 1
    if concentration < 500:
        status = 0
    else: status = 1
    timestamp = int(1000*time.time())
    print(time.strftime('%Y_%M_%D',time.localtime(time.time())))
    deviceType = 'laser'
    deviceIP = '192.168.20.102'

    infoAI = {
                "deviceID":deviceID,  
                "deviceIP":deviceIP,
                "deviceType":deviceType, 
                "correlationID": correlationID,  
                "timestamp":timestamp,
                "status": status, 
                "ptz": {
                        "horizontal":p,
                        "vertical":t,
                        "zoom":z,
                        },
                "concentration":concentration, 
                "Image":"",
                "irImage":"",
                }

    cam_info = copy.deepcopy(infoAI)    
    json_path = save_path+deviceID+save_img_path+'infoAI.json'
    print("save_img_path:",save_img_path,'\n',json_path)
    save_json_file(cam_info, json_path)
    print(timestamp)
    # save_json_file(cam_info, '/home/docker/SaveImage/laser1##fasdfearfa##/2022_12_28/UximZml9/infoAI.json')
    # if status == 0:
    #     with open(save_path+"/alarm.txt","a") as f:
    #         f.write('\r\n'+deviceID+'/'+save_img_path)

from test_skClient import SocketClient, recive_data_thread
import traceback
import time
import threading

class TEST():
    def __init__(self, ):
        print('## starting ##')
        self.if_send = 1
        self.sk = SocketClient('127.0.0.1', 8881)

    def run(self,):
        self.socket_send()
        self.monitor_socket()
        # self.init_socket()
        self.printsss() 
        
    def printsss(self, ):
        count = 0
        while 1:
            print(f'---------{count}')
            time.sleep(1)
            count+=1
            
    @recive_data_thread    
    def monitor_socket(self, ):
        while 1:
            if not self.if_send:
                print('---socket_send restarting---')
                time.sleep(2)
                self.sk.connect()
                # self.sk.reconnect()
                self.socket_send()
                self.if_send = 1
    
    @recive_data_thread
    def socket_send(self, ):
        print('## socket_send INIT')
        count = 0
        try:
            while self.if_send:
                self.sk.send('132456789_'+str(count))
                time.sleep(1)
                count+=1            
        except AttributeError as err:
            print('BAD ###### AttributeError reconnect')
            print(err)
            self.if_send = 0
            exit(0)
        except ConnectionResetError as ce:
            print('BAD ###### ConnectionResetError reconnect')
            print(ce)
            self.if_send = 0
            exit(0)
        except BrokenPipeError as be:
            print('BAD ###### BrokenPipeError reconnect')
            print(be)
            self.if_send = 0
            exit(0)
        except Exception as e:
            print("\n" + traceback.format_exc())
            self.if_send = 0
            exit(0)
            
if __name__ == '__main__':
    try:
        aa = TEST()
        aa.run()
    except Exception as err:
        print(err)
    finally:
        pass
