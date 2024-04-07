import numpy as np

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
    diou = iou - center_distance / (diagonal_distance + 1e-6)
    return diou

def diou_nms(bboxes, scores, threshold=0.5):
    # 将边界框和得分按照得分进行排序
    indices = np.argsort(scores)[::-1]
    bboxes = bboxes[indices]
    nms_bboxes = []
    while bboxes.shape[0] > 0:
        # 选择得分最高的框
        best_bbox = bboxes[0]
        if not any(np.array_equal(best_bbox, b) for b in nms_bboxes):
            nms_bboxes.append(best_bbox)
        if bboxes.shape[0] == 1:
            break
        # 计算剩余框与最佳框的DIoU值
        diou = np.array([DIou(best_bbox, bbox, wh=True) for bbox in bboxes[1:]])
        # 删除DIoU值大于阈值的框
        bboxes = bboxes[1:][diou < threshold]
    return np.array(nms_bboxes)

boxes =  np.array([[100,100,210,210],[250,250,420,420],
                   [220,220,320,330],[100,100,210,210],
                   [230,240,325,330],[220,230,315,340]
                   ]) 
scores = np.array([0.72, 0.8, 0.92, 0.72, 0.81, 0.9])
thresh = 0.7
keep_indices = diou_nms(boxes, scores, thresh)
print("Indices to keep:", np.where(keep_indices)[0])
