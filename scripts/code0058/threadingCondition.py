#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
1\判断线程是否释放锁资源, 其他线程再进行操作，
使用threading.Condition()对象
'''
import os
import time 
from threading import Thread

import threading
import time

class PeriodicTimer:
    def __init__(self, interval):
        self._interval = interval
        self._flag = 0
        self.condition_lock = threading.Condition()

    def start(self):
        t = threading.Thread(target=self.run)
        t.daemon = True
        t.start()

    def run(self):
        '''
        Run the timer and notify waiting threads after each interval
        '''
        while True:
            time.sleep(self._interval)
            with self.condition_lock:
                 self._flag ^= 1
                 self.condition_lock.notify_all()

    def wait_for_tick(self):
        '''
        Wait for the next tick of the timer
        '''
        with self.condition_lock:
            last_flag = self._flag
            while last_flag == self._flag:
                self.condition_lock.wait()

# Example use of the timer
ptimer = PeriodicTimer(5)
ptimer.start()

# Two threads that synchronize on the timer
def countdown(nticks):
    while nticks > 0:
        ptimer.wait_for_tick()
        print('T-minus', nticks)
        nticks -= 1

def countup(last):
    n = 0
    while n < last:
        ptimer.wait_for_tick()
        print('Counting', n)
        n += 1

threading.Thread(target=countdown, args=(10,)).start()
threading.Thread(target=countup, args=(5,)).start()

a = 0
if 1:
    a ^= 1
    print(a)





