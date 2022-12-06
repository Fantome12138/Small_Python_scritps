# -*- coding: utf-8 -*-

'''
指定目录下文件，获取其他文件夹下相同名称的文件

如：voc数据集中选择若干.jpg，从其他文件夹获取对应的.xml文件到新的文件夹
'''
import os
import os.path
import shutil

def GetFileNameAndExt(filename):
    (filepath, tempfilename) = os.path.split(filename)
    (shotname, extension) = os.path.splitext(tempfilename)
    return shotname, extension
 
source_dir = '/home/labels_tmnp/'
target_dir = '/data0/all_images/'
final_dir = '/home/images_tmp/'
 
# 1.将指定A目录下的文件名取出,并将文件名文本和文件后缀拆分出来
items = os.listdir(source_dir)  #得到文件夹下所有文件名称
s = []
for fileNum in items: #遍历文件夹
    if not os.path.isdir(fileNum): #判断是否是文件夹,不是文件夹才打开
        # print(fileNum)  #打印出文件名
        filename = os.path.join(source_dir, fileNum)
        # print('###', filename) #打印出文件路径
        (filepath, tempfilename) = os.path.split(filename); #将路径与文件名分开
        (shotname, extension) = os.path.splitext(tempfilename); #将文件名文本与文件后缀分开
        
# 2.将取出来的文件名文本与特定后缀拼接,再与路径B拼接,得到B目录下的文件
        temptargetname = '%s.jpg' % shotname
        # tempxmlname = '%s.xml' % shotname
        xmlname = os.path.join(target_dir, temptargetname)
        print(xmlname)
# 3.根据得到的xml文件名,将对应文件拷贝到指定目录C
        shutil.copy(xmlname, final_dir)

print('okk')
