import math
import numpy as np
import torch


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
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])
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
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])
    area1 = (xmax1-xmin1) * (ymax1-ymin1)
    area2 = (xmax2-xmin2) * (ymax2-ymin2)
    inter_area = np.max([0, xx2-xx1]) * np.max([0,yy2-yy1])
    iou = inter_area / (area1+area2-inter_area+1e-6)
    
    center_x1, center_y1 = (xmax1-xmin1)/2.0, (ymax1-ymin1)/2.0
    center_x2, center_y2 = (xmax2-xmin2)/2.0, (ymax2-ymin2)/2.0
    inter_diag = (center_x2-center_x1)**2 + (center_y2-center_y1)**2
    outer_diag = (xx2-xx1)**2 + (yy2-yy1)**2
    D = inter_diag / outer_diag
    diou = iou - D
    
    w1, h1 = xmax1-xmin1, ymax1-ymin1
    w2, h2 = xmax2-xmin2, ymax2-ymin2
    v = (4/np.pi**2) * (np.arctan(w1/h1) - np.arctan(w2/h2))**2
    a = v / ((1-iou) + v)
    ciou = diou - a*v
    return ciou


def rect_overlap(rect1, rect2): 
    # 返回两候选框重叠部分的坐标
    (x11, y11, x12, y12) = rect1     
    (x21, y21, x22, y22) = rect2     
    X1 = max(x11,x21)
    Y1 = max(y11,y21)
    X2 = min(x12,x22)
    Y2 = min(y12,y22)
    return [X1, Y1, X2, Y2]

def Giou(rec1, rec2):
    #分别是第一个矩形左右上下的坐标
    x1,x2,y1,y2 = rec1 
    x3,x4,y3,y4 = rec2
    iou = Iou(rec1,rec2)
    area_C = (max(x1,x2,x3,x4)-min(x1,x2,x3,x4))*(max(y1,y2,y3,y4)-min(y1,y2,y3,y4))
    area_1 = (x2-x1)*(y1-y2)
    area_2 = (x4-x3)*(y3-y4)
    sum_area = area_1 + area_2

    w1 = x2 - x1   #第一个矩形的宽
    w2 = x4 - x3   #第二个矩形的宽
    h1 = y1 - y2
    h2 = y3 - y4
    W = min(x1,x2,x3,x4)+w1+w2-max(x1,x2,x3,x4)    #交叉部分的宽
    H = min(y1,y2,y3,y4)+h1+h2-max(y1,y2,y3,y4)    #交叉部分的高
    Area = W*H    #交叉的面积
    add_area = sum_area - Area    #两矩形并集的面积

    end_area = (area_C - add_area)/area_C    # 闭包区域中不属于两个框的区域占闭包区域的比重
    giou = iou - end_area
    return giou

def DIou(box1, box2, wh=False):
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
    # DIoU = IoU - 中心距离/对角线距离
    diou = iou - center_distance / (diagonal_distance + 1e-6)

    return diou

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
    w1 = xmax1 - xmin1
    h1 = ymax1 - ymin1
    w2 = xmax2 - xmin2
    h2 = ymax2 - ymin2
    inter_area = (np.max([0, np.min([xmax1, xmax2])-np.max([xmin1, xmin2])])) * (np.max([0, np.min([ymax1, ymax2])-np.max([ymin1, ymin2])]))
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    iou = inter_area / (union_area + 1e-6)
    
    center_distance = np.power(center_x1 - center_x2, 2) + np.power(center_y1 - center_y2, 2)
    diagonal_distance = np.power(np.max([xmax1, xmax2])-np.min([xmin1, xmin2]), 2) + np.power(np.max([ymax1, ymax2])-np.min([ymin1, ymin2]), 2)
    
    v = 4 / (np.pi ** 2) * (np.arctan(w2 / (h2 + 1e-6)) - np.arctan(w1 / (h1 + 1e-6))) ** 2
    # np.errstate是一个上下文管理器，用于处理浮点数错误。如忽略除0的错误
    with np.errstate(divide='ignore',invalid='ignore'):
        alpha = v / (1 - iou + v)
    ciou = iou - (center_distance / (diagonal_distance + 1e-6) + alpha * v)
    
    return ciou


def bbox_overlaps_ciou(bboxes1, bboxes2):
    # Complete IoU，它在DIoU的基础上，还能同时考虑两个矩形的长宽比，也就是形状的相似性
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    cious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return cious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        cious = torch.zeros((cols, rows))
        exchange = True

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:],bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2],bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1+area2-inter_area
    u = (inter_diag) / outer_diag
    iou = inter_area / union
    with torch.no_grad():
        arctan = torch.atan(w2 / h2) - torch.atan(w1 / h1)
        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
        S = 1 - iou
        alpha = v / (S + v)
        w_temp = 2 * w1
    ar = (8 / (math.pi ** 2)) * arctan * ((w1 - w_temp) * h1)
    cious = iou - (u + alpha * ar)
    cious = torch.clamp(cious,min=-1.0,max = 1.0)
    if exchange:
        cious = cious.T
    return cious

