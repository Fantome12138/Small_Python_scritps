#!/usr/bin/env python3
# coding=UTF-8
import time
from functools import wraps
import logging
import logging.handlers


class loggings(object):
    
    def __init__(self, loggername=None,):
        self.mylogging = logging.getLogger(loggername)
        self.mylogging.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s[line:%(lineno)d]- %(levelname)s:  %(funcName)s - %(message)s")
        rotatingHandler = logging.handlers.TimedRotatingFileHandler(filename='/home/robot/vision_logs/main_log/main.log',
                                                                    when='MIDNIGHT', interval=1, backupCount=5)
        rotatingHandler.setFormatter(formatter)
        self.mylogging.addHandler(rotatingHandler)

    def getlog(self, ):
        return self.mylogging

mylogging = loggings('utils').getlog()


# @count_time装饰器: 写日志，计算函数运行时间
def CountTime(func):
	def decorate(*args, **kwargs):
		start_time = time.time()
		func(*args, **kwargs)
		end_time = time.time()
		mylogging.info(f'function |{func.__name__}| runing time: {round((end_time-start_time),4)}s')
	return decorate

# @logger装饰器: 写日志,记录函数输入输出数据
def ParamsLogger(func):
    @wraps(func)
    def log(*args, **kwargs):
        res = func(*args, **kwargs)
        mylogging.info(f"params: {func.__name__} {args} -> {res}")
        return func(*args, **kwargs)
    return log


@ParamsLogger
def bar(a,b):
    count = 0
    time.sleep(0.5)
    print(count)
    count += 1
    return a+b
   
c = bar(1,2)
print(c)
   