from PIL import Image
import matplotlib.pyplot as plt
import numpy as np 
import math

def BiLinear_interpolation(img,dstH,dstW):
    scrH,scrW,_=img.shape
    img=np.pad(img,((0,1),(0,1),(0,0)),'constant')
    retimg=np.zeros((dstH,dstW,3),dtype=np.uint8)
    for i in range(dstH):
        for j in range(dstW):
            scrx=(i+1)*(scrH/dstH)-1
            scry=(j+1)*(scrW/dstW)-1
            x=math.floor(scrx)
            y=math.floor(scry)
            u=scrx-x
            v=scry-y
            retimg[i,j]=(1-u)*(1-v)*img[x,y]+u*(1-v)*img[x+1,y]+(1-u)*v*img[x,y+1]+u*v*img[x+1,y+1]
    return retimg

im_path='../paojie.jpg'
image=np.array(Image.open(im_path))
image2=BiLinear_interpolation(image,image.shape[0]*2,image.shape[1]*2)
image2=Image.fromarray(image2.astype('uint8')).convert('RGB')
image2.save('out.png')



def bilinear_interpolation2(img,rate=1.1):
    h,w,c = img.shape
    # 计算放大后的尺寸
    new_h = int(h*rate)
    new_w = int(w*rate)
    # 将放大后的坐标映射到原始坐标范围
    h_s = np.linspace(0,h-1,new_h-1)
    w_s = np.linspace(0,w-1,new_w-1)
    # 创建一个空白的放大后尺寸的图像
    new_img = np.zeros([new_h,new_w,c],dtype=np.uint8)
    # 逐像素的进行计算
    for i in range(new_h-2):
        for j in range(new_w-2):
            for k in range(c):
                # 左上角坐标
                h0, w0 = int(h_s[i]), int(w_s[j])
                # 获得四个点的像素值
                q_00 = img[h0][w0][k]
                q_01 = img[h0][w0+1][k]
                q_10 = img[h0][w0][k]
                q_11 = img[h0+1][w0+1][k]
                # 使用公式计算映射点像素值
                new_img[i][j][k] = (h_s[i]-h0-1) * (w_s[j]-w0-1) * q_00 \
                                 + (h0+1-h_s[i]) * (w_s[j]-w0) * q_01 \
                                 + (h_s[i]-h0) * (w0+1-w_s[j]) * q_10 \
                                 + (h_s[i]-h0) * (w_s[j]-w0) * q_11 
    return new_img

