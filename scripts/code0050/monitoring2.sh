#!/bin/bash

#不需要手动 nohup python3 jing_app_run.py >/data/nohup.out/ 2>&1 & 启动,直接运行shell脚本

PROJECT=`basename $0 | awk -F '_' '{print $1}'`
APP_PATH="/data/app/$PROJECT"
APP_RUN_NAME=`ls $APP_PATH | grep py`
APP_PID=$(ps -ef | grep jing_app_run.py | grep -v grep | awk '{print $2}')
if [ -n "${APP_PID}" ]
then
kill -9 ${APP_PID}
sleep 5
fi
# 启动
cd ${APP_PATH}
nohup python3 ${APP_PATH}/${APP_RUN_NAME} > /data/logs/$PROJECT.out 2>&1 &