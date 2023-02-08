def Iou(box1, box2, wh=False):
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
    inter_area = (np.max([0, xx2-xx1])) * (np.max([0, yy2-yy1])) 
    iou = inter_area / (area1+area2-inter_area+1e-6)  
    return iou

###########################################
# 返回两候选框重叠部分的坐标
def rect_overlap(rect1, rect2): 
    (x11, y11, x12, y12) = rect1     
    (x21, y21, x22, y22) = rect2     
    X1 = max(x11,x21)
    Y1 = max(y11,y21)
    X2 = min(x12,x22)
    Y2 = min(y12,y22)
    return [X1, Y1, X2, Y2]