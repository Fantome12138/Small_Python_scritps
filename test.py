def check_online():
    """在下面的check_git_status、check_requirements等函数中使用
    检查当前主机网络连接是否可用
    """
    import socket  # 导入socket模块 可解决基于tcp和ucp协议的网络传输
    try:
        # 连接到一个ip 地址addr("1.1.1.1")的TCP服务上, 端口号port=443 timeout=5 时限5秒 并返回一个新的套接字对象
        socket.create_connection(("1.1.1.1", 443), 5)  # check host accessibility
        # 没发现什么异常, 连接成功, 有网, 就返回True
        return True
    except OSError:
        # 连接异常, 没网, 返回False
        return False
    
print(check_online())