def clip_coords(boxes, img_shape):
    """
    用在下面的xyxy2xywhn、save_one_boxd等函数中
    将boxes的坐标(x1y1x2y2 左上角右下角)限定在图像的尺寸(img_shape hw)内
    Clip bounding x1y1x2y2 bounding boxes to image shape (height, width)
    
    x的正坐标是向右, y的正坐标是向下
    """
    if isinstance(boxes, torch.Tensor):
        # .clamp_(min, max): 将取整限定在(min, max)之间, 超出这个范围自动划到边界上
        boxes[:, 0].clamp_(0, img_shape[1])  # x1
        boxes[:, 1].clamp_(0, img_shape[0])  # y1
        boxes[:, 2].clamp_(0, img_shape[1])  # x2
        boxes[:, 3].clamp_(0, img_shape[0])  # y2
    else:  # np.array
        boxes[:, 0].clip(0, img_shape[1], out=boxes[:, 0])  # x1
        boxes[:, 1].clip(0, img_shape[0], out=boxes[:, 1])  # y1
        boxes[:, 2].clip(0, img_shape[1], out=boxes[:, 2])  # x2
        boxes[:, 3].clip(0, img_shape[0], out=boxes[:, 3])  # y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    用在detect.py和test.py中  将预测坐标从feature map映射回原图
    将坐标coords(x1y1x2y2)从img1_shape缩放到img0_shape尺寸
    Rescale coords (xyxy) from img1_shape to img0_shape
    :params img1_shape: coords相对于的shape大小
    :params coords: 要进行缩放的box坐标信息 x1y1x2y2  左上角 + 右下角
    :params img0_shape: 要将coords缩放到相对的目标shape大小
    :params ratio_pad: 缩放比例gain和pad值   None就先计算gain和pad值再pad+scale  不为空就直接pad+scale
    
    用于detect.py中将预测坐标映射回原图
    """
    # ratio_pad为空就先算放缩比例gain和pad值 calculate from img0_shape
    if ratio_pad is None:
        # gain  = old / new  取高宽缩放比例中较小的,之后还可以再pad  如果直接取大的, 裁剪就可能减去目标
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        # wh padding  wh中有一个为0  主要是pad另一个
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]  # 指定比例
        pad = ratio_pad[1]  # 指定pad值

    # 因为pad = img1_shape - img0_shape 所以要把尺寸从img1 -> img0 就同样也需要减去pad
    # 如果img1_shape>img0_shape  pad>0   coords从大尺寸缩放到小尺寸 减去pad 符合
    # 如果img1_shape<img0_shape  pad<0   coords从小尺寸缩放到大尺寸 减去pad 符合
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    # 缩放scale
    coords[:, :4] /= gain
    # 防止放缩后的坐标过界 边界处直接剪切
    clip_coords(coords, img0_shape)
    return coords


def xyxy2xywh(x):
    """"
    用在detect.py和test.py中   
    操作最后, 将预测信息从xyxy格式转为xywh格式 再保存
    Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where x1y1=top-left, x2y2=bottom-right
    :params x: [n, x1y1x2y2] (x1, y1): 左上角   (x2, y2): 右下角
    :return y: [n, xywh] (x, y): 中心点  wh: 宽高
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    """
    用在test.py中 操作之前 转为xyxy才可以进行操作
    注意: x的正方向为右面   y的正方向为下面
    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where x1y1=top-left, x2y2=bottom-right
    :params x: [n, xywh] (x, y):
    :return y: [n, x1y1x2y2] (x1, y1): 左上角  (x2, y2): 右下角
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    """
    用在datasets.py的 LoadImagesAndLabels类的__getitem__函数、load_mosaic、load_mosaic9等函数中  
    将xywh(normalized) -> x1y1x2y2   (x, y): 中间点  wh: 宽高   (x1, y1): 左上点  (x2, y2): 右下点
    Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False):
    """
    用在datasets.py的 LoadImagesAndLabels类的__getitem__函数中
    将 x1y1x2y2 -> xywh(normalized)  (x1, y1): 左上点  (x2, y2): 右下点  (x, y): 中间点  wh: 宽高
    Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    """
    if clip:
        # 是否需要将x的坐标(x1y1x2y2)限定在尺寸(h, w)内
        clip_coords(x, (h, w))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    """
    用在datasets.py的load_mosaic和load_mosaic9函数中
    xy(normalized) -> xy
    Convert normalized segments into pixel segments, shape (n,2)
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * x[:, 0] + padw  # top left x
    y[:, 1] = h * x[:, 1] + padh  # top left y
    return y




