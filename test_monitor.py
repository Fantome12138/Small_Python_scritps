import os
import signal
import subprocess
import logging
import logging.handlers
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_MISSED, EVENT_JOB_EXECUTED
sched = BlockingScheduler()

'''
sudo vi /etc/rc.local 添加 su - nvidia -c "/home/nvidia/vision_logs/monitor.py" nvidia &
'''

class loggings(object):

    def __init__(self, loggername=None,):
        self.mylogging = logging.getLogger(loggername)
        self.mylogging.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s[line:%(lineno)d]- %(levelname)s:  %(funcName)s - %(message)s")
        rotatingHandler = logging.handlers.TimedRotatingFileHandler(filename='/Users/fantome/Library/CloudStorage/OneDrive-个人/Git/1_Github/Some_Python_Scripts/monitor_pyscripts.log',
                                                                    when='MIDNIGHT', interval=1, backupCount=5)
        rotatingHandler.setFormatter(formatter)
        self.mylogging.addHandler(rotatingHandler)

    def getlog(self, ):
        return self.mylogging

mylogging = loggings('monitor').getlog()

def get_process_id(name):
    child = subprocess.Popen(["pgrep","-f",name],stdout=subprocess.PIPE,shell=False)
    response = child.communicate()[0]
    return response

def start_sc(name, shell):
    pid = get_process_id(name)
    if not pid:
        restart_result = os.system(shell)
        if restart_result == 0:
            mylogging.info('---starting shell scripts %s---', name)
        else:
            mylogging.info('starting shell scripts failed! %s', name)
    else:
        mylogging.info('scripts %s exist', name)

def kill_sc(name, shell):
    cc = []
    pid = get_process_id(name)
    if not pid:
        start_sc(name, shell)
    else: 
        pid = pid.decode('utf-8').replace('\n', '#').split('#', -1)
        for it in pid:
            if it != '':
                cc.append(int(it))
                os.kill(int(it), signal.SIGKILL)  
        mylogging.info('KILLED pid {}'.format(cc))
        start_sc(name, shell)
        
        
def monitor():
    scripts = ["test_always.py"]
    path1 = ["/Users/fantome/Library/CloudStorage/OneDrive-个人/Git/1_Github/Some_Python_Scripts/"]
    for index, name in enumerate(scripts):
        if os.path.exists(path1[index]):
            shell = "nohup python3 "+str(path1[index])+name+" >/Users/fantome/Library/CloudStorage/OneDrive-个人/Git/1_Github/Some_Python_Scripts/output.log 2>&1 &"
            kill_sc(name, shell)
        else: pass

def main(time):
    sched._logger = mylogging
    sched.add_job(monitor, 'interval', seconds=int(time) , id='monitor')
    logging.getLogger('apscheduler.executors.default').setLevel(logging.WARNING)       
    sched.start()

if __name__ == '__main__':
    mylogging.info('----- go to monitor -----')
    # main(2)
    monitor()
