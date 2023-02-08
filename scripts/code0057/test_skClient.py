import socket
import time
import threading
from functools import wraps

def recive_data_thread(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.setDaemon(True)
        thread.start()
    return wrapper

class SocketClient(object):
    
    def __init__(self, ip, port):
        self.ip = ip 
        self.port = port
        self.buffrt_size = 1000
        self.connected = False
        self.sk = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.count = 0
        self.connect()
    
    def connect(self):
        print('waiting server...')
        while 1:
            try:
                self.sk = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
                self.sk.connect((self.ip, self.port))
                ret = str(self.sk.recv(self.buffrt_size), encoding="utf-8")
                if ret:
                    print('i am SocketClient: ', ret)
                    return 1
            except: pass
        
    def reconnect(self,):
        self.sk = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.sk.connect((self.ip, self.port))
        print('i am SocketClient, try reconnect')
        self.connected = True 
    
    def send(self, msg):
        self.sk.sendall(bytes(msg,encoding='utf-8'))
    
    def receive(self, ):
        return self.sk.recv(self.buffrt_size)
    
    def close(self, ):
        self.sk.shutdown(socket.SHUT_RDWR)
        self.sk.close()                 

    def run(self, ):
        time.sleep(1)
        i = 5
        try:    
            while i:
                if self.connected:
                    self.count+=1
                    self.send('hahahahahha')
                    msg = self.receive()
                    print(msg,'  --',self.count)
                    time.sleep(1)
                    i -= 1
        except Exception as err:
            self.connected = False
            print(err)
        finally:
            print('-----------------------')
            
 

