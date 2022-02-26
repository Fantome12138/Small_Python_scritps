# 剪裁图像区域  cropImg = image[c:d,a:b]

### opencv使用内置方法截取ROI区域 ###
import cv2

img = cv2.imread('C:\\roller3.jpg')
roi = cv2.selectROI(windowName="roi", img=img, showCrosshair=True, fromCenter=False)
x, y, w, h = roi
img_w, img_h= img.shape[0], img.shape[1]
cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(255,0,0), thickness=2)
cv2.imshow("roi", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


### opencv截取区域并保存 ###
import cv2
import os
i = 1
for root, dirs, files in os.walk('/home/Image'):
    for file in files:
        if ('.jpg' in file):
            path = os.path.join(root, file)
            x, y, w, h = 226, 536, 447, 318
            image = cv2.imread(path, 0)
            image = image[536:536+318, 226:226+447]
            filename = './roller' + str(i) + '.jpg'
            cv2.rectangle(img=image, pt1=(x, y), pt2=(x + w, y + h), color=(255,0,0), thickness=2)
            cv2.imwrite(filename, image)
            i += 1

### 1\指定图像位置的裁剪处理 ###
import os   
import cv2 
 
# 遍历指定目录，显示目录下的所有文件名
def CropImage4File(filepath,destpath):
    pathDir =  os.listdir(filepath)    # 列出文件路径中的所有路径或文件
    for allDir in pathDir:
        child = os.path.join(filepath, allDir)
        dest = os.path.join(destpath,allDir)
        if os.path.isfile(child):
        	image = cv2.imread(child) 
            sp = image.shape            #获取图像形状：返回【行数值，列数值】列表
            sz1 = sp[0]                 #图像的高度（行 范围）
            sz2 = sp[1]                 #图像的宽度（列 范围）
            #sz3 = sp[2]                #像素值由【RGB】三原色组成
            
            #你想对文件的操作
            a=int(sz1/2-64) # x start
            b=int(sz1/2+64) # x end
            c=int(sz2/2-64) # y start
            d=int(sz2/2+64) # y end
            cropImg = image[a:b,c:d]   #裁剪图像
            cv2.imwrite(dest,cropImg)  #写入图像路径
           
if __name__ == '__main__':
    filepath ='F:\\\maomi'             #源图像
    destpath='F:\\maomi_resize'        # resized images saved here
    CropImage4File(filepath,destpath)

### 2\批量处理—指定图像位置的裁剪 ###
"""
处理数据集 和 标签数据集的代码：（主要是对原始数据集裁剪）
    处理方式：分别处理
    注意修改 输入 输出目录 和 生成的文件名
    output_dir = "./label_temp"
    input_dir = "./label"
"""
import cv2
import os
import sys
import time


def get_img(input_dir):
    img_paths = []
    for (path,dirname,filenames) in os.walk(input_dir):
        for filename in filenames:
            img_paths.append(path+'/'+filename)
    print("img_paths:",img_paths)
    return img_paths


def cut_img(img_paths,output_dir):
    scale = len(img_paths)
    for i,img_path in enumerate(img_paths):
        a = "#"* int(i/1000)
        b = "."*(int(scale/1000)-int(i/1000))
        c = (i/scale)*100
        time.sleep(0.2)
        print('正在处理图像： %s' % img_path.split('/')[-1])
        img = cv2.imread(img_path)
        weight = img.shape[1]
        if weight>1600:                         # 正常发票
            cropImg = img[50:200, 700:1500]    # 裁剪【y1,y2：x1,x2】
            #cropImg = cv2.resize(cropImg, None, fx=0.5, fy=0.5,
                                 #interpolation=cv2.INTER_CUBIC) #缩小图像
            cv2.imwrite(output_dir + '/' + img_path.split('/')[-1], cropImg)
        else:                                        # 卷帘发票
            cropImg_01 = img[30:150, 50:600]
            cv2.imwrite(output_dir + '/'+img_path.split('/')[-1], cropImg_01)
        print('{:^3.3f}%[{}>>{}]'.format(c,a,b))

if __name__ == '__main__':
    output_dir = "../img_cut"           # 保存截取的图像目录
    input_dir = "../img"                # 读取图片目录表
    img_paths = get_img(input_dir)
    print('图片获取完成 。。。！')
    cut_img(img_paths,output_dir)

### 3\多进程（加快处理） ###
#coding: utf-8
"""
采用多进程加快处理。添加了在读取图片时捕获异常，OpenCV对大分辨率或者tif格式图片支持不好
处理数据集 和 标签数据集的代码：（主要是对原始数据集裁剪）
    处理方式：分别处理
    注意修改 输入 输出目录 和 生成的文件名
    output_dir = "./label_temp"
    input_dir = "./label"
"""
import multiprocessing
import cv2
import os
import time


def get_img(input_dir):
    img_paths = []
    for (path,dirname,filenames) in os.walk(input_dir):
        for filename in filenames:
            img_paths.append(path+'/'+filename)
    print("img_paths:",img_paths)
    return img_paths


def cut_img(img_paths,output_dir):
    imread_failed = []
    try:
        img = cv2.imread(img_paths)
        height, weight = img.shape[:2]
        if (1.0 * height / weight) < 1.3:       # 正常发票
            cropImg = img[50:200, 700:1500]     # 裁剪【y1,y2：x1,x2】
            cv2.imwrite(output_dir + '/' + img_paths.split('/')[-1], cropImg)
        else:                                   # 卷帘发票
            cropImg_01 = img[30:150, 50:600]
            cv2.imwrite(output_dir + '/' + img_paths.split('/')[-1], cropImg_01)
    except:
        imread_failed.append(img_paths)
    return imread_failed


def main(input_dir,output_dir):
    img_paths = get_img(input_dir)
    scale = len(img_paths)

    results = []
    pool = multiprocessing.Pool(processes = 4)
    for i,img_path in enumerate(img_paths):
        a = "#"* int(i/10)
        b = "."*(int(scale/10)-int(i/10))
        c = (i/scale)*100
        results.append(pool.apply_async(cut_img, (img_path,output_dir )))
        print('{:^3.3f}%[{}>>{}]'.format(c, a, b)) # 进度条（可用tqdm）
    pool.close()                        # 调用join之前，先调用close函数，否则会出错。
    pool.join()                         # join函数等待所有子进程结束
    for result in results:
        print('image read failed!:', result.get())
    print ("All done.")



if __name__ == "__main__":
    input_dir = "D:/image_person"       # 读取图片目录表
    output_dir = "D:/image_person_02"   # 保存截取的图像目录
    main(input_dir, output_dir)
