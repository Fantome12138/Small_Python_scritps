import numpy as np


def IOU(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    
    xx1 = np.max(xmin1, xmin2)
    xx2 = np.min(xmax1, xmax2)
    yy1 = np.max(ymin1, ymin2)
    yy2 = np.min(ymax1, ymax2)
    area1 = (xmax1-xmin1) * (ymax1-ymin1)
    area2 = (xmax2-xmin2) * (ymax2-ymin2)
    inter_area = np.max([0, xx2-xx1]) * np.max([0, yy2-yy1])
    iou = inter_area / (area1+area2-inter_area+1e-6)    
    
    center_x1, center_y1 = (xmax1-xmin1)/2.0, (ymax1-ymin1)/2.0
    center_x2, center_y2 = (xmax2-xmin2)/2.0, (ymax2-ymin2)/2.0
    inter_diag = (center_x2-center_x1)**2 + (center_y2-center_y1)**2
    outer_diag = (xx2-xx1)**2 + (yy2-yy1)**2
    D = inter_diag / outer_diag
    diou = iou - D
    
    w1, h1 = xmax1-xmin1, ymax1-ymin1
    w2, h2 = xmax2-xmin2, ymax2-ymin2
    v = (4/np.pi**2) * (np.arctan(w1/h1) - np.arctan(w2/h2))
    a = v / ((1-iou) + v)
    ciou = diou - a*v
    
    return ciou
    
    

def isPointInPolygon(point, rangelist):
    lnglist, latlist = [], []
    for i in range(len(rangelist)-1):
        lnglist.append(rangelist[i][0])
        latlist.append(rangelist[i][1])
    maxlng, minlng = max(lnglist), min(lnglist)
    maxlat, minlat = max(latlist), min(latlist)
    if (point[0]>maxlng) or (point[0]<minlng) or \
        (point[1]>maxlat) or (point[1]<minlat):
        return  False
    
    count = 0
    point1 = rangelist[0]
    for i in range(1, len(rangelist)):
        point2 = rangelist[i]
        # 点与多边形顶点重合
        if ((point[0]==point1[0]) and (point[1]==point1[1])) or \
            ((point[0]==point2[0]) and (point[1]==point2[1])):
            print('点与多边形顶点重合')
            return False

        # 判断线段两端点是否在射线两侧，不在肯定不相交
        if (point1[1]<point[1] and point2[1]>=point[1]) or \
            (point1[1]>=point[1] and point2[1]<point[1]):
            # 求线段与射线交点 再和lat比较
            point12lng = point2[0] - (point2[1]-point[1]) * (point2[0]-point1[0])/(point2[1]-point1[1])
            if point12lng == point[0]:
                return False
            if point12lng < point[0]:
                count += 1
        point1 = point2
    if count%2 == 0:
        return False
    else: return True
    

class MaxPolling(object):
    def __init__(self, kernel=(2,2), stride=2):
        self.kernel = kernel
        self.w_height = kernel[0]
        self.w_width = kernel[1]
        self.stride = stride
        
        self.x = None
        self.in_height, self.in_width = None, None
        self.out_height, self.out_width = None, None
        self.arg_max = None
    
    def __call__(self, x):
        self.x = x
        self.in_height = np.shape(x)[0]
        self.in_width = np.shape(x)[1]
        
        self.out_height = int((self.in_height-self.w_height) / self.stride) + 1
        self.out_width = int((self.in_width-self.w_width) / self.stride) + 1
        
        out = np.zeros((self.out_height, self.out_width))
        self.arg_max = np.zeros_like(out, detype=np.int32)
        
        for i in range(self.out_height):
            for j in range(self.out_width):
                start_i = i * self.stride
                start_j = j * self.stride
                end_i = start_i + self.w_height
                end_j = start_j + self.w_width
                out[i, j] = np.max(x[start_i:end_i, start_j:end_j])
                self.arg_max[i, j] = np.argmax(x[start_i:end_i, start_j:end_j])
        return out
    
    def backward(self, d_loss):
        dx = np.zeros_like(self.x)
        for i in range(self.out_height):
            for j in range(self.out_width):
                start_i = i * self.stride
                start_j = j * self.stride
                end_i = start_i + self.w_height
                end_j = start_j + self.w_width
                index = np.unravel_index(self.arg_max[i,j], self.kernel)
                dx[start_i:end_i, start_j:end_j][index] = d_loss[i,j]
        return dx
    
def conv_2d(input, kernel, stride):
    c, h, w = input.shape
    kernel_c, kernel_h, kernel_w = kernel.shape
    stride_h, stride_w = stride
    
    padding_h, padding_w = (kernel_h-1)//2, (kernel_w)//2
    padding_data = np.zeros([c, h+padding_h*2, w+padding_w*2])
    padding_data[:, padding_h:-padding_h, padding_w:-padding_w] = input
    
    out = np.zeros((h//stride_h, w//stride_w))
    for idx_h, i, in enumerate(range(0, h-kernel_h, stride_h)):
        for idx_w, j in enumerate(range(0, w-kernel_w, stride_w)):
            window = padding_data[:, i:i+kernel_h, j:j+kernel_w]
            out[idx_h, idx_w] = np.sum(window*kernel)
    return out


def onehot(targets, num):
    result = np.zeros((num, 10))
    for i in range(num):
        result[i][targets[i]] = 1
    return result



