#!/usr/bin/env python
# coding=UTF-8
import os
import glob 
from pathlib import Path  

current_path = os.path.abspath(__file__)
current_path = os.path.dirname(current_path)
print(current_path)

help_url = ''
prefix = ''
path = '/home/test/img'
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes

# 得到path路径下的所有图片的路径self.img_files，以yolov5代码为例
try:
    f = []  # image files
    for p in path if isinstance(path, list) else [path]:
        # 获取数据集路径path，包含图片路径的txt文件或者包含图片的文件夹路径
        # 使用pathlib.Path生成与操作系统无关的路径，因为不同操作系统路径的‘/’会有所不同
        p = Path(p)  # os-agnostic
        # 如果路径path为包含图片的文件夹路径
        if p.is_dir():  # dir
            # glob.glab: 返回所有匹配的文件路径列表  递归获取p路径下所有文件
            f += glob.glob(str(p / '**' / '*.*'), recursive=True)
            # f = list(p.rglob('**/*.*'))  # pathlib
        # 如果路径path为包含图片路径的txt文件
        elif p.is_file():  # file
            with open(p, 'r') as t:
                t = t.read().strip().splitlines()  # 获取图片路径，更换相对路径
                # 获取数据集路径的上级父目录  os.sep为路径里的分隔符（不同路径的分隔符不同，os.sep可以根据系统自适应）
                parent = str(p.parent) + os.sep
                f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
        else:
            raise Exception(f'{prefix}{p} does not exist')
    # 破折号替换为os.sep，os.path.splitext(x)将文件名与扩展名分开并返回一个列表
    # 筛选f中所有的图片文件
    img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in img_formats])
    print(img_files)
    # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in img_formats])  # pathlib
    assert img_files, f'{prefix}No images found'
except Exception as e:
    raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {help_url}')
