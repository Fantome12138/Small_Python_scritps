import cv2
import os
import colorsys
import random
from tqdm import tqdm

def get_n_hls_colors(num):
  hls_colors = []
  i = 0
  step = 360.0 / num
  while i < 360:
    h = i
    s = 90 + random.random() * 10
    l = 50 + random.random() * 10
    _hlsc = [h / 360.0, l / 100.0, s / 100.0]
    hls_colors.append(_hlsc)
    i += step

  return hls_colors

def ncolors(num):
  rgb_colors = []
  if num < 1:
    return rgb_colors
  hls_colors = get_n_hls_colors(num)
  for hlsc in hls_colors:
    _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
    r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
    rgb_colors.append([r, g, b])

  return rgb_colors

def convert(bbox,shape):
    x1 = int((bbox[0] - bbox[2] / 2.0) * shape[1])
    y1 = int((bbox[1] - bbox[3] / 2.0) * shape[0])
    x2 = int((bbox[0] + bbox[2] / 2.0) * shape[1])
    y2 = int((bbox[1] + bbox[3] / 2.0) * shape[0])
    return (x1,y1,x2,y2)

n = 80 # 类别数
# 获取n种区分度较大的rgb值
colors = ncolors(n)

images_list = os.listdir('images/') # 获取图片名列表
images_dir = 'images/' # 图片目录
labels_dir = 'labels/' # label目录
output_dir = 'showbbox/' # 输出图片目录

# import pdb;pdb.set_trace()
for img_id in tqdm(images_list):
    img = cv2.imread(images_dir + img_id) 
    # 判断后缀是为了排除隐藏文件.ipynb_checkpoints
    if img_id[-4:] != 'jpeg' and img_id[-3:] != 'jpg':
      continue
    shape = img.shape[0:2]
    txt_id = img_id.replace('jpeg', 'txt').replace('jpg', 'txt') 
    with open(labels_dir + txt_id) as r:
        lines = r.readlines()
        for line in lines:
            line = [float(i) for i in line.split(' ')] # 按空格划分并转换float类型
            label = int(line[0]) #获取类别信息
            bbox = line[1:] # 获取box信息
            (x1,y1,x2,y2) = convert(bbox,shape)
            # import pdb;pdb.set_trace()
            cv2.rectangle(img, (x1, y1), (x2, y2), (colors[label][2], colors[label][1], colors[label][0]), 3)
            # cv2.waitKey(0)
            cv2.putText(img, "{}".format(label),
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors[label], 2)

    # print(output_dir + img_id)
    cv2.imwrite(output_dir + img_id, img)
