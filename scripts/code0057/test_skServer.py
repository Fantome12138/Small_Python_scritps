import socket


class SocketServer():
    
    def __init__(self, port=8881):
        self.sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.buffer_size = 1000
        self.sk.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
        self.sk.bind(('127.0.0.1', port))
        self.sk.listen(10)
        self.wait_connect() 

    def wait_connect(self,):
        print('waiting for client...')
        self.conn, self.address = self.sk.accept()
        self.conn.sendall(bytes("connected",encoding="utf-8"))   

    def recv_message(self):
        res = ''
        print('recv_message start ...')
        try:
            res = str(self.conn.recv(self.buffer_size), encoding="utf-8")
        except Exception as e:
            print('### recv_message error ', e)
        if res != '':
            print(res)
        return res

    def send_message(self,s):
        self.conn.sendall(bytes(s, encoding='utf-8'))
    
    def send_data(self, data_to_send, ):
        c = self.conn.sendall( data_to_send)
        
    def run(self, ):
        count = 0
        while 1:
            try:
                res = str(self.conn.recv(self.buffer_size), encoding="utf-8")
                print(res, '::', count)
                count += 1
                self.send_data(bytes("### server get data ###",encoding="utf-8"))
            except: 
                self.wait_connect() 

server = SocketServer()
server.run()


