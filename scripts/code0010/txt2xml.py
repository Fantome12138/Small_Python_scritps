from xml.dom.minidom import Document
import os
import cv2

def makexml(txtPath, xmlPath, picPath): #读取txt路径，xml保存路径，数据集图片所在路径
        dict = {'0': "Worker",#字典对类型进行转换
                '1': "Static_crane",
                '2': "Hanging_head",
                '3': "Crane",
                '4': "Roller",
                '5': "Bulldozer",
                '6': "Excavator",
                '7': "Truck",
                '8': "Loader", 
                '9': "Pump_truck", 
                '10':"Concrete_mixer", 
                '11':"Pile_driving",
                '12':"Other_vehicle", 
                '13':"head",
                '14':"helmet",
                '15':"reflective_clothes",
                '16':"other_clothes",
                '17':"Well_drill",
                '18':"Rotary_cultivator",
                '19':"Tree_excavator",
                '20':"Tractor" 
                }
        files = os.listdir(txtPath)
        for i, name in enumerate(files):
          xmlBuilder = Document()
          annotation = xmlBuilder.createElement("annotation")  # 创建annotation标签
          xmlBuilder.appendChild(annotation)
          # txtFile=open(txtPath+name)
          txtFile = open(txtPath+name)
          txtList = txtFile.readlines()
          img = cv2.imread(picPath+name[0:-4]+".jpg")
          Pheight, Pwidth, Pdepth=img.shape
            
          for i in txtList:
             oneline = i.strip().split(" ")
             folder = xmlBuilder.createElement("folder")#folder标签
             folderContent = xmlBuilder.createTextNode("VOC2007")
             folder.appendChild(folderContent)
             annotation.appendChild(folder)

             filename = xmlBuilder.createElement("filename")#filename标签
             filenameContent = xmlBuilder.createTextNode(name[0:-4]+".jpg")
             filename.appendChild(filenameContent)
             annotation.appendChild(filename)

             size = xmlBuilder.createElement("size")  # size标签
             width = xmlBuilder.createElement("width")  # size子标签width
             widthContent = xmlBuilder.createTextNode(str(Pwidth))
             width.appendChild(widthContent)
             size.appendChild(width)
             height = xmlBuilder.createElement("height")  # size子标签height
             heightContent = xmlBuilder.createTextNode(str(Pheight))
             height.appendChild(heightContent)
             size.appendChild(height)
             depth = xmlBuilder.createElement("depth")  # size子标签depth
             depthContent = xmlBuilder.createTextNode(str(Pdepth))
             depth.appendChild(depthContent)
             size.appendChild(depth)
             annotation.appendChild(size)

             object = xmlBuilder.createElement("object")
             picname = xmlBuilder.createElement("name")
             nameContent = xmlBuilder.createTextNode(dict[oneline[0]])
             picname.appendChild(nameContent)
             object.appendChild(picname)
             pose = xmlBuilder.createElement("pose")
             poseContent = xmlBuilder.createTextNode("Unspecified")
             pose.appendChild(poseContent)
             object.appendChild(pose)
             truncated = xmlBuilder.createElement("truncated")
             truncatedContent = xmlBuilder.createTextNode("0")
             truncated.appendChild(truncatedContent)
             object.appendChild(truncated)
             difficult = xmlBuilder.createElement("difficult")
             difficultContent = xmlBuilder.createTextNode("0")
             difficult.appendChild(difficultContent)
             object.appendChild(difficult)
             bndbox = xmlBuilder.createElement("bndbox")
             xmin = xmlBuilder.createElement("xmin")
             mathData = int(((float(oneline[1]))*Pwidth+1)-(float(oneline[3]))*0.5*Pwidth)
             xminContent = xmlBuilder.createTextNode(str(mathData))
             xmin.appendChild(xminContent)
             bndbox.appendChild(xmin)
             ymin = xmlBuilder.createElement("ymin")
             mathData = int(((float(oneline[2]))*Pheight+1)-(float(oneline[4]))*0.5*Pheight)
             yminContent = xmlBuilder.createTextNode(str(mathData))
             ymin.appendChild(yminContent)
             bndbox.appendChild(ymin)
             xmax = xmlBuilder.createElement("xmax")
             mathData = int(((float(oneline[1]))*Pwidth+1)+(float(oneline[3]))*0.5*Pwidth)
             xmaxContent = xmlBuilder.createTextNode(str(mathData))
             xmax.appendChild(xmaxContent)
             bndbox.appendChild(xmax)
             ymax = xmlBuilder.createElement("ymax")
             mathData = int(((float(oneline[2]))*Pheight+1)+(float(oneline[4]))*0.5*Pheight)
             ymaxContent = xmlBuilder.createTextNode(str(mathData))
             ymax.appendChild(ymaxContent)
             bndbox.appendChild(ymax)
             object.appendChild(bndbox)

             annotation.appendChild(object)

          f = open(xmlPath+name[0:-4]+".xml", 'w')
          xmlBuilder.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
          f.close()

txtPath = ''
xmlPath = ''
picPath = ''
makexml(txtPath, xmlPath, picPath)



# OR

import os, sys
import glob
from PIL import Image
import argparse

def txtLabel_to_xmlLabel(classes_file,source_txt_path,source_img_path,save_xml_path):
    if not os.path.exists(save_xml_path):
        os.makedirs(save_xml_path)
    classes = open(classes_file).read().splitlines()
    print(classes)
    for file in os.listdir(source_txt_path):
        img_path = os.path.join(source_img_path,file.replace('.txt','.jpg')) #png to jpg
        img_file = Image.open(img_path)
        txt_file = open(os.path.join(source_txt_path,file)).read().splitlines()
        print(txt_file)
        xml_file = open(os.path.join(save_xml_path,file.replace('.txt','.xml')), 'w')
        width, height = img_file.size
        xml_file.write('<annotation>\n')
        xml_file.write('\t<folder>simple</folder>\n')
        xml_file.write('\t<filename>' + str(file) + '</filename>\n')
        xml_file.write('\t<size>\n')
        xml_file.write('\t\t<width>' + str(width) + ' </width>\n')
        xml_file.write('\t\t<height>' + str(height) + '</height>\n')
        xml_file.write('\t\t<depth>' + str(3) + '</depth>\n')
        xml_file.write('\t</size>\n')

        for line in txt_file:
            print(line)
            line_split = line.split(' ')
            x_center = float(line_split[1])
            y_center = float(line_split[2])
            w = float(line_split[3])
            h = float(line_split[4])
            xmax = int((2*x_center*width + w*width)/2)
            xmin = int((2*x_center*width - w*width)/2)
            ymax = int((2*y_center*height + h*height)/2)
            ymin = int((2*y_center*height - h*height)/2)

            xml_file.write('\t<object>\n')
            xml_file.write('\t\t<name>'+ str(classes[int(line_split[0])]) +'</name>\n')
            xml_file.write('\t\t<pose>Unspecified</pose>\n')
            xml_file.write('\t\t<truncated>0</truncated>\n')
            xml_file.write('\t\t<difficult>0</difficult>\n')
            xml_file.write('\t\t<bndbox>\n')
            xml_file.write('\t\t\t<xmin>' + str(xmin) + '</xmin>\n')
            xml_file.write('\t\t\t<ymin>' + str(ymin) + '</ymin>\n')
            xml_file.write('\t\t\t<xmax>' + str(xmax) + '</xmax>\n')
            xml_file.write('\t\t\t<ymax>' + str(ymax) + '</ymax>\n')
            xml_file.write('\t\t</bndbox>\n')
            xml_file.write('\t</object>\n')
        xml_file.write('</annotation>')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes_file', type=str, default="C:\\Users\\10619\\OneDrive\\103_1215_before17\\person.name")
    parser.add_argument('--source_txt_path', type=str, default="C:\\Users\\10619\\OneDrive\\103_1215_before17\\annotation\\")
    parser.add_argument('--source_img_path', type=str, default="C:\\Users\\10619\\OneDrive\\103_1215_before17\\")
    parser.add_argument('--save_xml_path', type=str, default="C:\\Users\\10619\\OneDrive\\103_1215_before17\\Done\\")
    opt = parser.parse_args()

    txtLabel_to_xmlLabel(opt.classes_file,opt.source_txt_path,opt.source_img_path,opt.save_xml_path)