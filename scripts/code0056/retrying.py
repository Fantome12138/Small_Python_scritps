import traceback
from func_timeout import func_set_timeout
import func_timeout
from retrying import retry

'''
1、通过func_timeout模块和retrying模块实现python函数超时重试
'''
from func_timeout import func_set_timeout
import time
import func_timeout
from retrying import retry
 
def retry_if_error(exception):
    print("---------------------------")
    return isinstance(exception, func_timeout.exceptions.FunctionTimedOut)
 
@retry(retry_on_exception=retry_if_error)
@func_set_timeout(3)
def task():
    while True:
        print('hello world')
        time.sleep(1)
        
           
'''
2、@retry
retry_on_exception:根据异常或函数的判断,再是否执行retry
stop_max_attempt_number: retry次数
wait_fixed: 每次retry的时间间隔
stop_max_delay:总的执行retry时长
'''

def retry_if_error(exception):
    print("---------------------------")
    return isinstance(exception, AssertionError)

@retry(retry_on_exception=retry_if_error,stop_max_attempt_number=2,\
        wait_fixed=1000,stop_max_delay=10000)
def test():
    errorcode = 111
    ec = 0
    # try:
    time.sleep(2)
    assert errorcode == 1
    if errorcode == 1:
        ec = 1
        print('errorcode:',errorcode)
        raise Exception('error 1')
    elif errorcode == 2:
        ec = 2
        print('errorcode:',errorcode)
        raise Exception('error 2')
    elif errorcode is not str:
        ec = 3
        raise ValueError('ValueError 1')
    # except Exception as err:
    #     print("\n" + traceback.format_exc())
    # ec = 999
    print(ec)
test()
