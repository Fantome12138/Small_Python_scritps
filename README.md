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

(scripts/code001/copyfile2.py) 
获取文件夹下所有指定格式文件，复制到其他文件夹

(scripts/code001/copyfile3.py)
yolo格式标注文件查找对应目标，将图像复制到指定文件夹 

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

(scripts/code005/randomdelete.py) 

按比例随机删除文件

code006 | [CPU压力测试](scripts/code006/cputest.py) | ✔️

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

1、计算两矩形框IoU & 计算两矩形框相交区域的坐标（重合区）
2、GIou计算
3、CIou计算
4、DIou计算

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

code0037 | [yolov5代码详细解析](scripts/code0037/yolov5.md) |  

代码详细解析，引用自[CSDN--满船清梦压星河HK]
https://blog.csdn.net/qq_38253797/category_11222727.html
其中代码被部分分解并进行测试 /scripts/test_code/..

[获取文件路径](/code0037/test_code/files.py)
得到path1路径下的所有文件的路径
/code0037/test_code/img2label_paths.py
根据path1路径获取path2下所有文件的路径

[检查label](/code0037/test_code/verify_image_label.py)
用于检查每一张图片和每一张label文件是否完好

[hsv空间变换](/code0037/test_code/augmeng_hsv.py)

[检查主机是否联网](/code0037/test_code/check_online.py)

[检查文件是否存在](/code37/test_code/check_file.py)
检查相关文件路径是否可以找到该文件，若无返回None，找到则返回本地匹配到的第一个文件名，若为网络文件则下载

[切图像](/code37/test_code/clip_corrds.py)
将bbox坐标限定在图像尺寸内，防止出界
and 将坐标coords(x1y1x2y2)从img1_shape尺寸缩放到img0_shape尺寸
and xyxy2xywh  xywh2xyxy
and xywhn2xyxy、xyxy2xywhn、xyn2xy
xywhn2xyxy是将xywh(normalized) -> x1y1x2y2；xyxy2xywhn是将x1y1x2y2 -> xywh(normalized)；xyn2xy是将xy(normalized) -> xy

[Mosaic数据增强](/code37/test_code/Mosaic.py)

[NMS](/code37/test_code/non_max_suppression.py)

[计算测试数据集mAP](scripts/code0037/mAP/compute_mAP.py) | ✔️
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

[yolo自带val.py计算mAP](scripts/code0037/None) | ✔️
运行val.py获取mAP

[一些能用到的脚本](scripts/code0037/test_code/use.py)


code0038 | [判断点是否在多边形内](scripts/code0038/points.py) | ✔️
判断一个点是否在多边形区域内
           [计算点到直线的距离](scripts/code0038/(point2line.py) | 
判断两线是否相交

code0039 | [opencv提取指定颜色](scripts/code0039/Twog_r_b.py) | ✔️

寻找图像中指定颜色（绿色），并将其抠出来或计算指定颜色的像素总数

code0040 | [利用python多进程程或多线程的方式获取数据](scripts/code0040/mutiprocess.py) | ✔️

code0041 | [python获取linux系统信息](scripts/code0041/get_sysinfo.py) | ✔️

code0042 | [json数据字段过滤](scripts/code0042/jsonfilter.py) | ✔️

code0043 | [多属性排序funtools](scripts/code0043/funtools.py) | ✔️

code0044 | [Python 中循环语句速度对比](scripts/code0044/looptest.py) | ✔️

code0045 | [Python 列表的交集、并集、差集](scripts/code0045/test.py) | ✔️

code0046 | [超时处理函数timeout](scripts/code0046/timeout.py) | ✔️

scripts/code0046/timeout2.py （推荐）
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

code0059 | [flac](scripts/code0059/ncm2flac.py) | ✔️

ncm格式音乐转flac格式；改路径，flac文件会直接生成在改路径下

code0060 | [find circle](scripts/code0060/find_circle.py) | ✔️

opencv方法寻找圆形

code0061 | [find EulerAngles](scripts/code0061/EulerAngles.py) | ✔️

[欧拉角、旋转向量和旋转矩阵的相互转换](https://www.jianshu.com/p/5e130c04a602)

code0062 | [find QR code](scripts/code0062/find_qr.py) | ✔️

使用opencv，通过检测轮廓方式定位qrcode并抠出码

code0063 | [MaxPooling2D](scripts/code0063/MaxPooling2D.py) | ✔️

使用np,复现torch中的maxpooling函数; 卷积后，池化后尺寸计算公式：
(图像尺寸-卷积核尺寸 + 2*填充值)/步长+1

$out(N_i, C_{out_j}) = bias(C_{out_j})+\sum_{k=0}^{c_{in}-1}weight(C_{out_j}, k)*input(N_i,k)$ 

code0064 | [backprogram numpy](scripts/code0064/bp_mnist/fc_mnist.py) | ✔️

1、利用numpy实现神经网络,前向反向传播，mnist数据。bp_mnist/fc_mnist.py
https://zhuanlan.zhihu.com/p/86593676

2、纯numpy实现CNN模块（LeNet5）  
(scripts/code0064/numpy-realizes-CNN-master/) |
https://zhuanlan.zhihu.com/p/296592264

3、(scripts/code0064/mnist_cnn.py)
通过使用numpy来搭建一个基础的包含卷积层、池化层、全连接层和Softmax层的卷积神经网络，并选择relu作为我们的激活函数，选择多分类交叉熵损失函数，最后使用了mnist数据集进行了训练和测试。

4、(scripts/code0064/conv.py)
实现numpy卷积

5、（scripts/code0064/python_conv.py）
原生python实现卷积神经网络

https://blog.csdn.net/qq_43409114/article/details/105187448?spm=1001.2014.3001.5502 解析反向传播算法
https://zhuanlan.zhihu.com/p/447113449 前向传播（forward）和反向传播（backward）

code0065 | [BatchNormalization numpy](scripts/code0065/BatchNormalization1.py) | ✔️

numpy实现 BN

code0066 | [NMS numpy](scripts/code0066/NMS.py) | ✔️

1、将所有的boxes按照置信度从小到大排序，然后从boxes中删除置信度最大的box
2、将剩下的boxes与置信度最大的box，分别计算iou，去掉iou大于阈值(iou_threshold)的boxes
3、重复1，2直到索引为空

https://zhuanlan.zhihu.com/p/80902998 CUDA版本

anchor 如何映射回原图 (scripts/code0066/anchor.py)

code0067 | [focal loss &常见损失函数](scripts/code0067/focal_loss.py) | ✔️

numpy 实现cross entropy loss 和 focal loss 和常见损失函数

Logistic Regression:

使用Python实现一个逻辑回归模型，并使用梯度下降来最小化交叉熵损失函数(使用sigmoid激活)
(scripts/code0067/logistic_Regression.py)

code0068 | [bilinear interpolation](scripts/code0068/bilinear_interpolation.py) | ✔️

实现图像双线性插值

code0069 | [常见激活函数](scripts/code0069/activation_func.py) | ✔️

Sigmiod、Tanh、Relu、PReLU、Swish、Mish

code0070 | [编辑距离计算](scripts/code0070/Levenshtein_Distance.py) | ✔️

python计算两文本编辑距离，其内包含去除字符串标点和英文的方法

Python比较文本相似度的7种方法
https://blog.csdn.net/SpinMeRound/article/details/107465022


code0071 | [K-means py实现](scripts/code0071/kmeans.py) | ✔️

及yolo中使用的k-means方法聚类anchor尺寸

code0072 | [中值滤波 py实现](scripts/code0072/medianBlur.py) | ✔️

numpy实现中值、均值滤波，高斯滤波

1、均值滤波：

通过取像素周围邻域内的像素值的平均值来替换该像素的值，从而达到平滑图像的效果

2、高斯滤波器：

将中心像素周围的像素按照高斯分布加权平均进行平滑化。这样的（二维）权值通常被称为卷积核（kernel）或者滤波器（filter）

但是，由于图像的长宽可能不是滤波器大小的整数倍，因此我们需要在图像的边缘补0 。这种方法称作Zero Padding

权值g（卷积核）要进行归一化操作 $G(x)=\frac{1}{2\pi \eth^2}e^{-\frac{x^2+y^2}{2\eth ^2}}$

高斯掩膜的求解与位置(x,y)无关，因为在计算过程中x,y被抵消掉了,所以求一个kerne模板即可


code0073 | [增广数据，复制-粘贴](scripts/code0073/demo.py) | ✔️

https://blog.csdn.net/zengwubbb/article/details/113061776
https://blog.csdn.net/oyezhou/article/details/111696577
数据集增广，适用于小目标，对抠图下来的小目标随机粘贴至图像并保存其bbbox

code0074 | [md文件中查找修改内容](scripts/code0074/demo.py) | ✔️

code0075 | [手撕 transformer ](scripts/code0075/test_transformer.py) | ✔️

code0076 | [v7 onnx推理](scripts/code0076/detect_onnx_no_nms.py) | ✔️
           [v7 onnx推理 MNS](scripts/code0076/detect_onnx_with_nms.py)

带NMS的没测过

code0077 | [yolo计算模型参数量](scripts/code0077/count_params.py) | ✔️

yolo计算模型参数量和算力

code0078 | [yolo数据集格式转coco format](scripts/code0078/txt2json.py) | ✔️

yolo数据集txt格式label转成coco使用的json格式。
1、split_train_val，将yolo数据分成train val（默认2:8）
2、change_diff_class，根据要求改labels
3、txt2json，生成train val的json
4、concate_json，合并不同数据集的json

code0079 | [凸四边形IOU计算](scripts/code0079/convex_quadrilaterals_iou.py) | ✔️
https://www.hbblog.cn/OCR%E7%9B%B8%E5%85%B3/%E4%BB%BB%E6%84%8F%E5%87%B8%E5%9B%9B%E8%BE%B9%E5%BD%A2iou%E7%9A%84%E8%AE%A1%E7%AE%97/

1.py 旋转矩形的坐标转换
2.py 判断线段相交 可见code0038
3.py 判断点是否在凸多边形内部 可见code0038
4.py 凸包算法
5.py 计算多边形面积
convex_quadrilaterals_iou.py 凸四边形IOU计算

code0080 | [手撕resnet](scripts/code0080/resnet.py) | ✔️

code0081 | [计算两矩阵欧式距离](scripts/code0081/euclidean_distance.py) | ✔️

$\rho= \sqrt{(x_2-x_1)^2+(y_2-y_1)^2}$

$|x|=\sqrt{x^2_2+y^2_2}$

其中 $\rho$ 是点间的欧氏距离，|x|是点(x2,y2)到原点的欧式距离

code0082 | [线性卷积、互相关、自相关 Python实现](scripts/code0082/conv.py) | ✔️

code0083 | [手撕adam、adamW](scripts/code0083/adam.py) | ✔️

code0084 | [手撕分组注意力 groupAttention](scripts/code0084/groupAttention.py) | ✔️

Grouped Query Attention 是一种注意力机制，它在自然语言处理（NLP）中被用来增强模型对输入序列的理解能力。这种机制通常在Transformer模型的变体中使用，通过将查询（queries）、键（keys）和值（values）进行分组，可以提高模型的效率和性能。
在实现Grouped Query Attention时，我们通常会遵循以下步骤：

初始化参数：定义模型中的参数，包括用于分组的参数和用于计算注意力分数的参数。

准备输入数据：将输入序列分割成多个组，每个组包含一定数量的元素。这可以通过预先定义的规则或者根据输入数据的特点来完成。

计算注意力分数：对于每个组，计算查询与键之间的点积，然后应用softmax函数来获取注意力权重。

应用注意力权重：将注意力权重应用到对应的值上，然后将所有组的结果合并起来。

输出结果：最后，输出经过注意力机制处理后的序列。

code0085 | [手撕 SVD 奇异值分解](scripts/code0085/svd.py) | ✔️

奇异值分解（Singular Value Decomposition，SVD）是线性代数中一种重要的矩阵分解方法。在很多应用领域，如信号处理、统计学、机器学习等，SVD都有着广泛的应用。对于一个 $m*×*n$ 的矩阵 A，SVD可以表示为：$A=U\sum V^T$ 其中：

- U 是一个 m×m 的单位正交矩阵（即 $U^TU=I$）。
- Σ 是一个 m×n 的对角矩阵，其对角线上的元素称为奇异值，是 A 的非负实数。
- V 是一个 n×n 的单位正交矩阵（即 $VV^T=I$）。

下面是手动实现SVD的步骤：

1. 计算矩阵 A 的协方差矩阵 $A^TA$。
2. 求解 $A^TA$ 的特征值和特征向量。
3. 根据特征值计算奇异值，它们是特征值的平方根。
4. 计算矩阵 V，它是 $A^TA$ 特征向量组成的矩阵。
5. 计算矩阵 U，它是 AV（这里 V 是经过适当的转置以匹配奇异值的顺序）。

code0086 | [手撕 SVM 支持向量机](scripts/code0086/svm.py) | ✔️

SVM用于分类和回归分析。核心思想是找到一个超平面，以最大化不同类别之间的边界（或称为间隔）来对数据进行分类。

首先，定义SVM的目标函数和优化问题。SVM的训练过程可以通过解决一个凸二次规划问题来实现，其目标是最大化间隔的同时最小化分类错误。这个问题可以通过拉格朗日乘子法转换为对偶问题，并使用序列最小优化（Sequential Minimal Optimization，SMO）算法或者梯度下降等方法求解。

code0087 | [最近邻|双线性插值](scripts/code0087/interpolation.py) | ✔️

code0088 | [](scripts/code0088/.py) | ✔️




