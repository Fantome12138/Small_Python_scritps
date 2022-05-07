import contextlib
import signal
import time

class timeout(contextlib.ContextDecorator):
    """没用到  代码中都是使用库函数自己定义的timeout 没用用这个自定义的timeout函数
    设置一个超时函数 如果某个程序执行超时  就会触发超时处理函数_timeout_handler 返回超时异常信息
    并没有用到  这里面的timeout都是用python库函数实现的 并不需要自己另外写一个
    使用: with timeout(seconds):  sleep(10)   或者   @timeout(seconds) decorator
    dealing with wandb login-options timeout issues as well as check_github() timeout issues
    """
    def __init__(self, seconds, *, timeout_msg='', suppress_timeout_errors=True):
        self.seconds = int(seconds)   # 限制时间
        self.timeout_message = timeout_msg  # 报错信息
        self.suppress = bool(suppress_timeout_errors)

    def _timeout_handler(self, signum, frame):
        # 超时处理函数 一旦超时 就在seconds后发送超时信息
        print(self.timeout_message)
        raise TimeoutError(self.timeout_message)

    def __enter__(self):
        # signal.signal: 设置信号处理的函数_timeout_handler
        # 执行流进入with中会执行__enter__方法 如果发生超时, 就会触发超时处理函数_timeout_handler 返回超时异常信息
        signal.signal(signal.SIGALRM, self._timeout_handler)  # Set handler for SIGALRM
        # signal.alarm: 设置发送SIGALRM信号的定时器
        signal.alarm(self.seconds)  # start countdown for SIGALRM to be raised

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 执行流离开 with 块时(没有发生超时), 则调用这个上下文管理器的__exit__方法来清理所使用的资源
        signal.alarm(0)  # Cancel SIGALRM if it's scheduled
        if self.suppress and exc_type is TimeoutError:  # Suppress TimeoutError
            return True

@timeout(seconds=2, timeout_msg='bad', suppress_timeout_errors=True)
def a(src):
    time.sleep(1)
    print(src)
    return 10

c = a('233')
print(c)