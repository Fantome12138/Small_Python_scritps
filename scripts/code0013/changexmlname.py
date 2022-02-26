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