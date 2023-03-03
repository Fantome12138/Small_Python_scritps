# Small_Python_Scripts

说明：下面都是俺常用的python小脚本，作为备份。(很多代码是网上现成的，但链接很多都找不到了，就这样吧)

## Learning Road ⛳️

**Annotation:**

\- ✔️  ***\*: Basic\****

\- ✏️  ***\*: Attention\****

\- ❣️  ***\*: Important\****

No.   | Description  | Annotation

:--------:  |  :--------:  |  :--------:

code001 | [根据当前文件名批量复制](scripts/code001/copyfile.py)  | ✔️

根据指定目录下文件，获取其他文件夹下相同名称文件，voc数据集中选择若干.jpg，从其他文件夹获取对应的.xml文件到新的文件夹

code002 | [两列表排序保持对应关系](scripts/code002/sort2list.py)  | ✔️

两个列表元素排序后保持元素对应关系

code003 | [文件重命名](scripts/code003/renamefile.py)  | ✔️

文件批量重命名脚本

code004 | [视频抽帧为图片](scripts/code004/video2img.py) | ✔️
           [图片整合成视频](scripts/code004/img2video.py) |

读取视频.mp4，将其抽帧转换成单幅图像保存
读取图片序列，将其转换成视频保存

code005 | [随机复制文件](scripts/code005/randomcopy.py) | ✔️

随机复制文件到指定文件夹

code006 | [CPU压力测试](scripts/code006/cputest.py) | ✔️

测试CPU性能

code007 | [cv2生成缩略图](scripts/code007/generateThumbnails.py) | ✔️

code008 | [json转txt格式](scripts/code008/json2txt.py) | ✔️

code009 | [opencv剪切ROI](scripts/code009/cv2cutROI.py) | ✔️

code0010 | [txt格式转xml](scripts/code0010/txt2xml.py) | ✔️
           [xml格式转txt](scripts/code0010/xml2txt.py) | ✔️

code0011 | [统计xml目标种类数量](scripts/code0011/countxml.py) | ✔️
         | [统计txt目标种类数量](scripts/code0011/counttxt.py) | ✔️
         | [统计数据集GT大中小个数](scripts/code0011/count_bboxSML.py) | ✔️
         局限:统计真实标注框大小，仅有96*96，32*32等，针对640^2的图片，自己的的数据集要改统计框的尺寸


根据xml文件统计目标种类以及数量

code0012 | [计算xml目标信息](scripts/code0012/countxmlobj.py) | ✔️

根据xml文件统计目标的平均长度、宽度、面积以及每一个目标在原图中的占比

code0013 | [修改xml文件中类别名称](scripts/code0013/changexmlname.py) | ✔️

包括修改xml节点内容

code0014 | [批量修改图像尺寸](scripts/code0014/resize.py) | ✔️

code0015 | [批量生成空的xml文件](scripts/code0015/generatexml.py) | ✔️

用于yolo的负样本生成

code0016 | [Python批量创建文件夹](scripts/code0016/folder.py) | ✔️

code0017 | [Python批量创建txt](scripts/code0016/txt.py) | ✔️

code0018 | [标注原始图像](scripts/code0018/drawbbox.py) | ✔️

批量将标注数据在原始图片中画出来&保存xml以便微调
         | [yolo格式画框](scripts/code0018/yolo_drawbbox.py) | ✔️
         | [coco格式画框](scripts/code0018/coco_drawbbox.py) | ✔️

code0019 | [测试生成bbox](scripts/code0019/drawbbox.py) | ✔️

推理测试图像，得到xml文件，以便手动调整bbox   for tolov5-6.0

code0020 | [yolo中文标签](scripts/code0020/label.py) | ✔️

YOLOv5图像识别显示中文标签

code0021 | [计算gps两点坐标距离](scripts/code0021/gps.py) | ✔️

code0022 | [常用queue队列脚本](scripts/code0022/queue.py) | ✔️

包括常用 Queue脚本 \ deque双边队列脚本 \ 队列多线程Multithreading

code0023 | [cv2写视频](scripts/code0023/videowriter.py) | ✔️

opencv读取图片并保存成视频

code0024 | [cv2视频格式转换](scripts/code0024/transvideo.py) | ✔️

视频格式转换，例 avi --> mp4

code0025 | [cv2读取海康视频流](scripts/code0025/hik.py) | ✔️

code0026 | [带时间戳的相机连续拍照](scripts/code0026/Monocular_cam.py) | ✔️

带有微秒级时间戳的单、双目相机连续拍照程序

code0027 | [几种读取图片保存的方法](scripts/code0027/readimg.py) | ✔️

code0028 | [读txt温度矩阵转图像](scripts/code0028/matrix2img.py) | ✔️

根据从热红外设备读取的温度矩阵（保存为txt）将其转换为伪彩图像

code0029 | [cv鼠标获取像素坐标](scripts/code0029/getcoordinates.py) | ✔️

OpenCV 鼠标点击获取像素坐标并写入txt文件，用于可见光、热红外手动标定工作(提取特征点)

code0030 | [读取txt文件](scripts/code0030/txt.py) | ✔️

python从txt文件中逐行读取数据

code0031 | [cv图像特征的提取](scripts/code0031/Feature_extraction.py) | ✔️

使用opencv提取图像特征：AKAZE、ORB    [Feature_extraction.py]
使用opencv对提取的特征进行筛选、匹配   [Filter_feature.py]

code0032 | [从rosbag读取图片](scripts/code0032/rosbag.py) | ✔️

code0033 | [IoU计算](scripts/code0033/IoU.py) | ✔️

计算两矩形框IoU & 计算两矩形框相交区域的坐标（重合区）

code0034 | [ArUco二维码](scripts/code0034/aruco.py) | ✔️

使用aruco标记创建和检测

code0035 | [图像格式转换](scripts/code0035/img_format_trans.py) | ✔️

code0036 | [自适应缩放图片](scripts/code0036/letterbox.py) | ✔️

yolov5自适应缩放图片，32的倍数缩放，空缺部分自动填充--选自/ultralytics/yolov5

scripts/code0036/resize_img.py
python等比例缩放图片

scripts/code0036/S2L_img.py
python 将小图放入较大的白色或黑色背景图片

scripts/code0036/dilate.py
opencv提取指定颜色区域及图像腐蚀、膨胀简单介绍

scripts/code0036/noise_img.py
对图像进行增强操作(旋转，缩放，剪切，平移，加噪声)

scripts/code0036/windows_img.py
Python对图片进行滑动窗提取局部区域

scripts/code0036/plot_point.py
Python在图片上绘制指定半径的圆

scripts/code0036/findContours.py
Python图片查找轮廓、多边形拟合、最小外接矩形操作实例

scripts/code0036/frame_diff.py
帧差法得到运动背景图像(简单差分方法)

## code0037 | [yolov5代码详细解析](scripts/code0037/yolov5.md) |  

代码详细解析，引用自[CSDN--满船清梦压星河HK]
https://blog.csdn.net/qq_38253797/category_11222727.html
其中代码被部分分解并进行测试 /scripts/test_code/..

/code0037/test_code/files.py
得到path1路径下的所有文件的路径

/code0037/test_code/img2label_paths.py
根据path1路径获取path2下所有文件的路径

/code0037/test_code/verify_image_label.py
用于检查每一张图片和每一张label文件是否完好

/code0037/test_code/augmeng_hsv.py
hsv空间变换

/code0037/test_code/check_online.py
检查当前主机是否联网了

/code37/test_code/check_file.py
检查相关文件路径是否可以找到该文件，若无返回None，找到则返回本地匹配到的第一个文件名，若为网络文件则下载

/code37/test_code/clip_corrds.py
将bbox坐标限定在图像尺寸内，防止出界
and 将坐标coords(x1y1x2y2)从img1_shape尺寸缩放到img0_shape尺寸
and xyxy2xywh  xywh2xyxy
and xywhn2xyxy、xyxy2xywhn、xyn2xy
xywhn2xyxy是将xywh(normalized) -> x1y1x2y2；xyxy2xywhn是将x1y1x2y2 -> xywh(normalized)；xyn2xy是将xy(normalized) -> xy

/code37/test_code/non_max_suppression.py
NMS(非极大值抑制)
.  
         | [计算测试数据集mAP](scripts/code0037/mAP/compute_mAP.py) | ✔️
         注：
         在yolo文件夹下创建./data_test
         并创建/Annotations_manual/    放置xml
               /JPEGImages_manual/    放置图像
               /predictions_manual/
               /cachedir_manual/
               /class_txt_manual/
         在cfg_mAP.py中修改配置(测试图片等)
         运行detect_eval_class_txt.py，获得测试图片的cls_ap.pkl
         运行mAP_line.py获得每个类别的mAP曲线
.
         | [yolo自带val.py计算mAP](scripts/code0037/None) | ✔️
         运行val.py获取mAP
         -
         | [一些能用到的脚本](scripts/code0037/test_code/use.py)


code0038 | [判断点是否在多边形内](scripts/code0038/points.py) | ✔️
判断一个点是否在多边形区域内
           [计算点到直线的距离](scripts/code0038/(point2line.py) | 


code0039 | [opencv提取指定颜色](scripts/code0039/Twog_r_b.py) | ✔️

code0040 | [利用python多进程程或多线程的方式获取数据](scripts/code0040/mutiprocess.py) | ✔️

code0041 | [python获取linux系统信息](scripts/code0041/get_sysinfo.py) | ✔️

code0042 | [json数据字段过滤](scripts/code0042/jsonfilter.py) | ✔️

code0043 | [多属性排序funtools](scripts/code0043/funtools.py) | ✔️

code0044 | [Python 中循环语句速度对比](scripts/code0044/looptest.py) | ✔️

code0045 | [Python 列表的交集、并集、差集](scripts/code0045/test.py) | ✔️

code0046 | [超时处理函数timeout](scripts/code0046/timeout.py) | ✔️

设置一个超时函数 如果某个程序执行超时  就会触发超时处理函数_timeout_handler 返回超时异常信息

code0047 | [打印或logging保存输出加颜色（好看）](scripts/code0047/color.py) | ✔️

将输出的开头和结尾加上颜色，使命令行输出显示会更加好看

code0048 | [强制停止线程](scripts/code0048/stopthread.py) | ✔️

code0049 | [切分图像并保留xml标记](scripts/code0049/crop_img.py) | ✔️

按步长切分图像，并保留每幅切分后图像含有的xml标记；问题：部分重叠区域目标bbox的xml会丢失。

code0050 | [监控python脚本是否运行](scripts/code0050/monitoring.py) | ✔️
         | [shell监控python脚本是否运行](scripts/code0050/monitoring.sh) | ✔️

monitoring.py使用python监控某py脚本是否运行
monitoring.sh使用shell监控某py脚本是否运行,否则后台启动脚本

code0051 | [py设置logging](scripts/code0051/pylogging.py) | ✔️

创建loggings类，多py脚本可创建log实例，保存至同一.log文件

code0052 | [yolov5自动标注数据](scripts/code0052/autolabel.py) | ✔️

linux环境下使用已有模型自动化标注新数据，生成xml文件

参考使用https://github.com/WangRongsheng/KDAT/tree/main/autoLabel

code0053 | [copy_paste数据增强](scripts/code0053/copy_paste.py) | ✔️
           针对小目标的数据增强，随机复制小目标到训练图像里
         | [数据增强反转图像及label](scripts/code0053/filpimgs.py) | ✔️
         | [数据增强简单增强亮度对比度](scripts/code0053/color.py) | ✔️
         

code0054 | [语义分割评判指标计算](scripts/code0054/judge.py) | ✔️

参考：https://blog.csdn.net/sinat_29047129/article/details/103642140

code0055 | [统计文件夹下各源代码文件行数](scripts/code0055/count_line.py) | ✔️

结果保存至.txt文件

code0056 | [常用装饰器](scripts/code0056/decorators.py) | ✔️
            装饰器：计算函数运行时长；记录函数输入输出数据
           [func_timeout和retrying](scripts/code0056/retrying.py) | ✔️
            实现函数超时重试 & 函数异常重试
            使用包func_timeout & retrying

code0057 | [python socket](scripts/code0057/test_socket.py) | ✔️

socket服务端、客户端，两方均实现断开重连

code0058 | [threading](scripts/code0058/threadingCondition.py) | ✔️

python多线程操作, 具体内容见Py3CookBook.md --> 第十二章：并发编程

实现使用wait()方法释放锁，并阻塞程序直到其他线程调用notify()或者notify_all()方法唤醒，然后wait()方法重新获取锁，
类似于event.wait()方式，但看起来更高级。

code0059 | [flac](scripts/code0059/ncm2flac_new.py) | ✔️

ncm格式音乐转flac格式

code0060 | [find circle](scripts/code0060/find_circle.py) | ✔️

opencv方法寻找圆形

code0061 | [find EulerAngles](scripts/code0061/EulerAngles.py) | ✔️

[欧拉角、旋转向量和旋转矩阵的相互转换](https://www.jianshu.com/p/5e130c04a602)

code0062 | [find QR code](scripts/code0062/find_qr.py) | ✔️

使用opencv，通过检测轮廓方式定位qrcode并抠出码

