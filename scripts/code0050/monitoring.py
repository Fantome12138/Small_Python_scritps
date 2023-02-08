'''
使用python监控python进程是否在运行
'''
import os
 
# execute command, and return the infomation. 执行命令并返回命令输出信息
def execCmd(cmd):
    r = os.popen(cmd)
    text = r.read()
    r.close()
    return text
 
def doSomething():
    print("start Running")
 
if __name__ == '__main__':
    #ps -ef是linux查看进程信息指令，|是管道符号导向到grep去查找特定的进程,最后一个|是导向grep过滤掉grep进程：因为grep查看程序名也是进程，会混到查询信息里
    programIsRunningCmd="ps -aux |grep listen_mqtt.py"    
    programIsRunningCmdAns = execCmd(programIsRunningCmd)    #调用函数执行指令，并返回指令查询出的信息
    ansLine = programIsRunningCmdAns.split('\n')    #将查出的信息用换行符‘\b’分开
    print(ansLine)
    #判断如果返回行数>2则说明python脚本程序已经在运行，打印提示信息结束程序，否则运行脚本代码doSomething()
    if len(ansLine) > 2:
        print("programName have been Running")
    else:  
        doSomething()

