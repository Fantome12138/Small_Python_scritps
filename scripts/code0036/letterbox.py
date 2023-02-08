import cv2 
import random
import numpy as np

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    # 调整图片大小，达到32的最小倍数

    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):  # [height, width]
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)   选择最小的缩放系数
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    """
       缩放(resize)到输入大小img_size的时候，如果没有设置上采样的话，则只进行下采样
       因为上采样图片会让图片模糊，对训练不友好影响性能。
    """
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))      # [width, height]
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle 最小矩形填充
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch  直接resize为img_size大小，任由图片拉伸压缩
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
    # 图像两边需要填充的宽度
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # 进行填充
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)



img = cv2.imread('/home/test/img/00014.jpg')
print(img.shape)
img0, ratio, d = letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True)
cv2.imwrite('/home/test/x.jpg', img0)
print(img0.shape)
print(ratio, d)

