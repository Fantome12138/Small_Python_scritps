#!/bin/bash

project1='/home/vision_proxy_gw/vision_logs/listen_mqtt.py'

for Pro in $project1

do

PythonPid=`ps -ef | grep $Pro | grep -v grep | wc -l `

echo $Pro
if [ $PythonPid -eq 0 ];
        then
        echo "`date "+%Y-%m-%d %H:%M:%S"`:$Pro is not running" >> /home/vision_proxy_gw/vision_logs/python.log

        cd /home/vision_proxy_gw/vision_logs/

        nohup python  $Pro >output 2>&1 &

        echo "`date "+%Y-%m-%d %H:%M:%S"`:$Pro is starting" >> /home/vision_proxy_gw/vision_logs/python.log
        sleep 5
        CurrentPythonPid=`ps -ef | grep $Pro | grep -v grep | wc -l`
        if [ $CurrentPythonPid -ne 0 ];
        then
        echo "`date "+%Y-%m-%d %H:%M:%S"`:$Pro is running" >> /home/vision_proxy_gw/vision_logs/python.log
        fi
fi
done
