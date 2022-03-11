#!/usr/bin/env python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib

'''图像尺寸由温度矩阵大小而定'''

f = open("C:\\Users\\2022.txt")
val_list = f.readlines()  # 288
lists =[]
for string in val_list:
    string = string.split('\t')  # 385
    lists.append(string[0:384])
    a = np.array(lists)
    print(a.shape)
    a = a.astype(float)
a = np.array(a).reshape(288, 384)

#####
#interpolation=nearest  插值
#origin=upper 镜像
#cmap=plt.cm.      jet\hsv\hot\cool\spring\summer\autumn\winter\gray\bone\copper\pink\lines
#####
# plt.imshow(a,interpolation='nearest', cmap=plt.cm.hot, origin='upper')
plt.imshow(a,interpolation='None',cmap=plt.cm.gray,origin='upper')

# cmap_color_list = ['Accent','Accent_r','Blues','Blues_r','BrBG','BrBG_r','BuGn','BuGn_r','BuPu','BuPu_r','CMRmap','CMRmap_r',\
# 'Dark2','Dark2_r','GnBu','GnBu_r','Greens','Greens_r','Greys','Greys_r','OrRd','OrRd_r','Oranges','Oranges_r',\
# 'PRGn','PRGn_r','Paired','Paired_r','Pastel1','Pastel1_r','Pastel2','Pastel2_r','PiYG','PiYG_r','PuBu','PuBuGn',\
# 'PuBuGn_r','PuBu_r','PuOr','PuOr_r','PuRd','PuRd_r','Purples','Purples_r','RdBu','RdBu_r','RdGy','RdGy_r','RdPu',\
# 'RdPu_r','RdYlBu','RdYlBu_r','RdYlGn','RdYlGn_r','Reds','Reds_r','Set1','Set1_r','Set2','Set2_r','Set3','Set3_r',\
# 'Spectral','Spectral_r','Wistia','Wistia_r','YlGn','YlGnBu','YlGnBu_r','YlGn_r','YlOrBr','YlOrBr_r','YlOrRd','YlOrRd_r',\
# 'afmhot','afmhot_r','autumn','autumn_r','binary','binary_r','bone','bone_r','brg','brg_r','bwr','bwr_r','cividis',\
# 'cividis_r','cool','cool_r','coolwarm','coolwarm_r','copper','copper_r','cubehelix','cubehelix_r','flag','flag_r',\
# 'gist_earth','gist_earth_r','gist_gray','gist_gray_r','gist_heat','gist_heat_r','gist_ncar','gist_ncar_r','gist_rainbow',\
# 'gist_rainbow_r','gist_stern','gist_stern_r','gist_yarg','gist_yarg_r','gnuplot','gnuplot2','gnuplot2_r','gnuplot_r',\
# 'gray','gray_r','hot','hot_r','hsv','hsv_r','inferno','inferno_r','jet','jet_r','magma',\
# 'magma_r','nipy_spectral','nipy_spectral_r','ocean','ocean_r','pink','pink_r','plasma','plasma_r',\
# 'prism','prism_r','rainbow','rainbow_r','seismic','seismic_r','spring','spring_r','summer',\
# 'summer_r','tab10','tab10_r','tab20','tab20_r','tab20b','tab20b_r','tab20c','tab20c_r','terrain','terrain_r',\
# 'twilight','twilight_r','twilight_shifted','twilight_shifted_r','viridis','viridis_r','winter','winter_r']

# 温标  shrink=0.8
# plt.colorbar()
# plt.xticks(())
# plt.yticks(())
# plt.show() 
#保存图片
matplotlib.image.imsave('C:\\Users\\2022.jpg', a)




# ###### csv文件
# lists = []
# data = []
# with open(self.csvpath.text(),'r',encoding='gbk',errors='ignore') as file:
#     for string in file:
#         data.append(string.rstrip('\n').split(',')[1:])
# lists = data[2:]#此时list中存放的都为有用数据

# num+=1
# for k in range(0,len(lists),348):
#      s = lists[k:k+348]
#      num += 1
#      self.progressBar_2.setValue(num)
#      self.label_8.setText(str(num))
#      a = np.array(s)
#      list1 = []
#      for i in range(0, 348):  # 194184
#          for j in range(464):
#              list1.append((round(float(a[i][j]),1)))#将温度数据保存一位小数
#      martix = np.array(list1).reshape(348, len(a[0]))# 形成图像矩阵

# plt.imshow(martix,vmax=800,vmin=150,interpolation='nearest',cmap=plt.cm.gnuplot,origin='upper')#cmap是自定义温度图标
# #保存
# plt.savefig(文件路径+文件名,bbox_inches="tight", pad_inches=0)
