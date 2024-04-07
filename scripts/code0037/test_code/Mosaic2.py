import numpy as np

def mosaic(imgs, labels):
    h, w, _ = imgs[0].shape  # 获取待处理图片的高和宽
    # Mosaic图像中心的位置
    center_x = int(np.random.uniform(w // 2, w // 2 * 3))
    center_y = int(np.random.uniform(h // 2, h // 2 * 3))
    mosaic_img = np.empty((2 * h, 2 * w, 3), dtype=np.uint8)  # 初始化mosaic图像
    mosaic_labels = []
    for i, img, label in zip(range(4), imgs, labels):
        # 对每个图像进行操作
        h_i, w_i, _ = img.shape
        # 分四种情况处理每个角的图像
        if i == 0:  # top-left
            x1, y1, x2, y2 = max(center_x - w_i, 0), max(center_y - h_i, 0), \
                                 center_x, center_y  # 计算此图像在mosaic图像中的位置
            mosaic_img[y1:y2, x1:x2] = img[(h_i - y2 + y1):, (w_i - x2 + x1):]  # 将图像放在对应位置

        elif i == 1:  # top-right
            x1, y1, x2, y2 = center_x, max(center_y - h_i, 0), min(center_x + w_i, w * 2), center_y
            mosaic_img[y1:y2, x1:x2] = img[(h_i - y2 + y1):, 0:(x2 - x1)]

        elif i == 2:  # bottom-left
            x1, y1, x2, y2 = max(center_x - w_i, 0), center_y, center_x, min(h * 2, center_y + h_i)
            mosaic_img[y1:y2, x1:x2] = img[0:(y2 - y1), (w_i - x2 + x1):]

        elif i == 3:  # bottom-right
            x1, y1, x2, y2 = center_x, center_y, min(center_x + w_i, w * 2), \
                            min(h * 2, center_y + h_i)
            mosaic_img[y1:y2, x1:x2] = img[0:(y2 - y1), 0:(x2 - x1)]

        # 调整标签的坐标
        padw = x1 - max(center_x - w_i, 0)
        padh = y1 - max(center_y - h_i, 0)

        # 调整bbox坐标，注意这里假设bbox格式为[x_center, y_center, w, h]，并且为归一化坐标
        if label.size > 0:
            label[:, [0, 2]] = label[:, [0, 2]] * w_i / w + padw / w
            label[:, [1, 3]] = label[:, [1, 3]] * h_i / h + padh / h
            mosaic_labels.append(label)

    if len(mosaic_labels) > 0:
        mosaic_labels = np.concatenate(mosaic_labels, 0)
        mosaic_labels[:, [0, 2]] = np.clip(mosaic_labels[:, [0, 2]], 0, 2)
        mosaic_labels[:, [1, 3]] = np.clip(mosaic_labels[:, [1, 3]], 0, 2)

    return mosaic_img, mosaic_labels
