import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

def Iou(box1, box2, wh=False):
    if wh == False:
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
    else:
        xmin1, ymin1 = int(box1[0]-box1[2]/2.), int(box1[1]-box1[3]/2.)
        xmax1, ymax1 = int(box1[0]+box1[2]/2.), int(box1[1]+box1[3]/2.)
        xmin2, ymin2 = int(box2[0]-box2[2]/2.), int(box2[1]-box2[3]/2.)
        xmax2, ymax2 = int(box2[0]+box2[2]/2.), int(box2[1]+box2[3]/2.)
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.max([xmax1, xmax2])
    yy2 = np.max([ymax1, ymax2])
    area1 = (xmax1-xmin1) * (ymax1-ymin1) # 第一个矩形框的面积
    area2 = (xmax2-xmin2) * (ymax2-ymin2) # 第二个矩形框的面积
    inter_area = (np.max([0, xx2-xx1])) * (np.max([0, yy2-yy1]))
    union_area = area1 + area2 - inter_area
    iou = inter_area / (union_area+1e-6)
    return iou

def GIou(box1, box2, wh=False):
    if wh == False:
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
    else:
        xmin1, ymin1 = int(box1[0]-box1[2]/2.), int(box1[1]-box1[3]/2.)
        xmax1, ymax1 = int(box1[0]+box1[2]/2.), int(box1[1]+box1[3]/2.)
        xmin2, ymin2 = int(box2[0]-box2[2]/2.), int(box2[1]-box2[3]/2.)
        xmax2, ymax2 = int(box2[0]+box2[2]/2.), int(box2[1]+box2[3]/2.)
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    inter_area = np.max([0, xx2 - xx1]) * np.max([0, yy2 - yy1])
    union_area = area1 + area2 - inter_area
    iou = inter_area / (union_area + 1e-6)
    # Calculate enclosing box 最小闭包区域
    enc_xmin = np.min([xmin1, xmin2])
    enc_ymin = np.min([ymin1, ymin2])
    enc_xmax = np.max([xmax1, xmax2])
    enc_ymax = np.max([ymax1, ymax2])
    enc_area = (enc_xmax - enc_xmin) * (enc_ymax - enc_ymin)
    # Calculate GIOU
    giou = iou - (enc_area - union_area) / (enc_area + 1e-6)
    return giou

def CIou(box1, box2, wh=False):
    if wh == False:
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
    else:
        xmin1, ymin1 = int(box1[0]-box1[2]/2.0), int(box1[1]-box1[3]/2.0)
        xmax1, ymax1 = int(box1[0]+box1[2]/2.0), int(box1[1]+box1[3]/2.0)
        xmin2, ymin2 = int(box2[0]-box2[2]/2.0), int(box2[1]-box2[3]/2.0)
        xmax2, ymax2 = int(box2[0]+box2[2]/2.0), int(box2[1]+box2[3]/2.0)
    center_x1 = (xmin1 + xmax1) / 2.0
    center_y1 = (ymin1 + ymax1) / 2.0
    center_x2 = (xmin2 + xmax2) / 2.0
    center_y2 = (ymin2 + ymax2) / 2.0
    # 计算框的宽度和高度
    w1 = xmax1 - xmin1
    h1 = ymax1 - ymin1
    w2 = xmax2 - xmin2
    h2 = ymax2 - ymin2
    # IoU计算
    inter_area = (np.max([0, np.min([xmax1, xmax2])-np.max([xmin1, xmin2])])) * (np.max([0, np.min([ymax1, ymax2])-np.max([ymin1, ymin2])]))
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    iou = inter_area / (union_area + 1e-6)
    # 中心点距离的平方
    center_distance = np.power(center_x1 - center_x2, 2) + np.power(center_y1 - center_y2, 2)
    # 对角线距离的平方
    diagonal_distance = np.power(np.max([xmax1, xmax2])-np.min([xmin1, xmin2]), 2) + np.power(np.max([ymax1, ymax2])-np.min([ymin1, ymin2]), 2)
    import math
    with torch.no_grad():
        arctan = torch.atan(w2 / h2) - torch.atan(w1 / h1)
        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
        S = 1 - iou
        alpha = v / (S + v)
        w_temp = 2 * w1
    ar = (8 / (math.pi ** 2)) * arctan * ((w1 - w_temp) * h1)
    u = center_distance / diagonal_distance
    cious = iou - (u + alpha * ar)
    return cious




    
    
    
    



box1 = [10, 20, 50, 60]  # [xmin, ymin, xmax, ymax]
box2 = [30, 40, 70, 80]

iou_value = DIou(box1, box2)
print(f"IOU between box1 and box2: {iou_value:.4f}")
