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
 
source_dir = '/Users/fantome/Downloads/新浦化学/数据集/xinpu_new1020/images'
target_dir = '/Users/fantome/Downloads/新浦化学/数据集/xinpu_new1020/labels'
final_dir = '/Users/fantome/Downloads/新浦化学/数据集/xinpu_new1020/new_labels'
 
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
        temptargetname = '%s.txt' % shotname
        # tempxmlname = '%s.xml' % shotname
        xmlname = os.path.join(target_dir, temptargetname)
        print(xmlname)
# 3.根据得到的xml文件名,将对应文件拷贝到指定目录C
        try:
            shutil.copy(xmlname, final_dir)
        except Exception as err:
            print(err)
            continue

print('okk')
