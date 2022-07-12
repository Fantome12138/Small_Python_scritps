import time
import inspect
import threading
import ctypes


def wait_Joint():
    c = True
    while c:
        print('111')
        time.sleep(1)
        
def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")
    
def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)


try:
    thread_machine_wait = threading.Thread(target=wait_Joint)
    print('Start thread')
    thread_machine_wait.start()
    time.sleep(5)
    stop_thread(thread_machine_wait)
    print('Stop thread')
except Exception as err:
    print('ERROR !  %s', err)
