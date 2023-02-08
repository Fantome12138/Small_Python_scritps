# 修改xml文件中的目标的名字
import os
import sys
import glob
from xml.etree import ElementTree as ET
# 批量读取Annotations下的xml文件
# per = ET.parse(r'C:\Users\rockhuang\Desktop\Annotations\000003.xml')
xml_dir = r'/home/Desktop/VOCdevkit/VOC2018/Annotations'
xml_list = glob.glob(xml_dir + '/*.xml')
for xml in xml_list:
    print(xml)
    per = ET.parse(xml)
    p = per.findall('/object')
    for oneper in p:  # 找出person节点
        child = oneper.getchildren()[0]  # 找出person节点的子节点
        if child.text == 'PinNormal':    # 需要修改的名字
            child.text = 'normal bolt'   # 修改成什么名字
        if child.text == 'PinDefect':    # 需要修改的名字
            child.text = 'defect bolt-1' # 修改成什么名字
    per.write(xml)
    print(child.tag, ':', child.text)


'''修改xml节点内容'''
import os
import xml.dom.minidom
import xml.etree.ElementTree
 
xmldir = '/home/xml/'
for xmlfile in os.listdir(xmldir):
    xmlname = os.path.splitext(xmlfile)[0]
 
    # read the xml file
    dom = xml.dom.minidom.parse(os.path.join(xmldir, xmlfile))
    root = dom.documentElement
 
    # obtain the filename label pair and give it a new value
    root.getElementsByTagName('filename')[0].firstChild.data = xmlname + '.jpg'
    root.getElementsByTagName('path')[0].firstChild.data = '/home/dulingwen/Music/jpg/' + xmlname + '.jpg'
    root.getElementsByTagName('width')[0].firstChild.data = '1920'
    root.getElementsByTagName('height')[0].firstChild.data = '1080'
    xml_specific = xmldir + xmlfile
    with open(xml_specific,'w') as fh:
        dom.writexml(fh)
        
#######################
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
 
import os
import xml.etree.ElementTree as ET
 
origin_ann_dir = '/home/YangziData/all_xml/'# 设置原始标签路径为 Annos
new_ann_dir = '/home/YangziData/new_xml/'# 设置新标签路径 Annotations
for dirpaths, dirnames, filenames in os.walk(origin_ann_dir):   # os.walk游走遍历目录名
  for filename in filenames:
    print("process...")
    if os.path.isfile(r'%s%s' %(origin_ann_dir, filename)):   # 获取原始xml文件绝对路径，isfile()检测是否为文件 isdir检测是否为目录
      origin_ann_path = os.path.join(r'%s%s' %(origin_ann_dir, filename))   # 如果是，获取绝对路径（重复代码）
      new_ann_path = os.path.join(r'%s%s' %(new_ann_dir, filename))
      tree = ET.parse(origin_ann_path)  # ET是一个xml文件解析库，ET.parse（）打开xml文件。parse--"解析"
      root = tree.getroot()   # 获取根节点
      for object in root.findall('object'):   # 找到根节点下所有“object”节点
        name = str(object.find('name').text)  # 找到object节点下name子节点的值（字符串）
    # 如果name等于str，则删除该节点
        if (name in ["pump", "pump_gauge"]):
          root.remove(object)
 
    # 如果name等于str，则修改name
        if(name in ["other_light"]):
          object.find('name').text = "person"
 
    # 检查是否存在labelmap中没有的类别
      for object in root.findall('object'):
        name = str(object.find('name').text)
        if not (name in ["chepai","chedeng","chebiao"]):
            print(filename + "------------->label is error--->" + name)
      tree.write(new_ann_path)#tree为文件，write写入新的文件中。
