import signal
import time
import functools
import sys, traceback

class TimeoutError(Exception):
    def __init__(self, err_msg):
        self.err_msg = err_msg
        pass
    
def timeout(sec):
    """
    timeout decorator
    :param sec: function raise TimeoutError after ? seconds
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):

            def _handle_timeout(signum, frame):
                # err_msg = f'Function {func.__name__} timed out after {sec} seconds'
                err_msg = 'Function timed out'
                raise TimeoutError(err_msg)

            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(sec)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapped_func
    return decorator


class ma(object):
    def __init__(self):
        pass
    @timeout(2)
    def slow_func(self,):
        time.sleep(3)
        return 1
    
def func2():
    try:
        func = ma()
        a = func.slow_func()
    except TimeoutError:
        print("func2333\n" + traceback.format_exc())
        a = 0
        raise Exception('func2 error')
    print('Done func2', a)

def func1():
    try:
        func2()
    except Exception as e:
        print("\n" + traceback.format_exc())
    print(sys.platform)
    
func1()


