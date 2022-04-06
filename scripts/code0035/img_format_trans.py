# -*- coding: utf-8 -*-
from PCV.tools.imtools import get_imlist  # 导入PCV模块
from PIL import Image
import os
import pickle

'''PIL可以将图片保存问很多不同的图像格式。下面的例子从文件名列表（filelist）中读取所有的图像文件，并转换成 JPEG 格式：'''
 
filelist = get_imlist('C:/Users/avg')  # 获取文件夹下的图片文件名

imlist = open('C:/Users/imlist.txt', 'wb')  # 将获取的图片文件列表保存到imlist.txt中
 
pickle.dump(filelist, imlist)  # 序列化
imlist.close()
 
 
for infile in filelist:
  outfile = os.path.splitext(infile)[0] + ".png"
  if infile != outfile:
    try:
      Image.open(infile).save(outfile)
    except IOError:
      print ("cannot convert", infile)


