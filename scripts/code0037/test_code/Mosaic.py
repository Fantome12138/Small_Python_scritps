from PIL import Image, ImageDraw
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import random

def load_mosaic(self, index):
    """用在LoadImagesAndLabels模块的__getitem__函数 进行mosaic数据增强
    将四张图片拼接在一张马赛克图像中  loads images in a 4-mosaic
    :param index: 需要获取的图像索引
    :return: img4: mosaic和随机透视变换后的一张图片  numpy(640, 640, 3)
             labels4: img4对应的target  [M, cls+x1y1x2y2]
    """
    # labels4: 用于存放拼接图像（4张图拼成一张）的label信息(不包含segments多边形)
    # segments4: 用于存放拼接图像（4张图拼成一张）的label信息(包含segments多边形)
    labels4, segments4 = [], []
    s = self.img_size  # 一般的图片大小
    # 随机初始化拼接图像的中心点坐标  [0, s*2]之间随机取2个数作为拼接图像的中心坐标
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
    # 从dataset中随机寻找额外的三张图像进行拼接 [14, 26, 2, 16] 再随机选三张图片的index
    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    # 遍历四张图像进行拼接 4张不同大小的图像 => 1张[1472, 1472, 3]的图像
    for i, index in enumerate(indices):
        # load image   每次拿一张图片 并将这张图片resize到self.size(h,w)
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left  原图[375, 500, 3] load_image->[552, 736, 3]   hwc
            # 创建马赛克图像 [1472, 1472, 3]=[h, w, c]
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)   w=736  h = 552  马赛克图像：(x1a,y1a)左上角 (x2a,y2a)右下角
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            # 计算截取的图像区域信息(以xc,yc为第一张图像的右下角坐标填充到马赛克图像中，丢弃越界的区域)  图像：(x1b,y1b)左上角 (x2b,y2b)右下角
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            # 计算截取的图像区域信息(以xc,yc为第二张图像的左下角坐标填充到马赛克图像中，丢弃越界的区域)
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            # 计算截取的图像区域信息(以xc,yc为第三张图像的右上角坐标填充到马赛克图像中，丢弃越界的区域)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            # 计算截取的图像区域信息(以xc,yc为第四张图像的左上角坐标填充到马赛克图像中，丢弃越界的区域)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        # 将截取的图像区域填充到马赛克图像的相应位置   img4[h, w, c]
        # 将图像img的【(x1b,y1b)左上角 (x2b,y2b)右下角】区域截取出来填充到马赛克图像的【(x1a,y1a)左上角 (x2a,y2a)右下角】区域
        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        # 计算pad(当前图像边界与马赛克边界的距离，越界的情况padw/padh为负值)  用于后面的label映射
        padw = x1a - x1b   # 当前图像与马赛克图像在w维度上相差多少
        padh = y1a - y1b   # 当前图像与马赛克图像在h维度上相差多少

        # labels: 获取对应拼接图像的所有正常label信息(如果有segments多边形会被转化为矩形label)
        # segments: 获取对应拼接图像的所有不正常label信息(包含segments多边形也包含正常gt)
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            # normalized xywh normalized to pixel xyxy format
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
        labels4.append(labels)      # 更新labels4
        segments4.extend(segments)  # 更新segments4

    # Concat/clip labels4 把labels4（[(2, 5), (1, 5), (3, 5), (1, 5)] => (7, 5)）压缩到一起
    labels4 = np.concatenate(labels4, 0)
    # 防止越界  label[:, 1:]中的所有元素的值（位置信息）必须在[0, 2*s]之间,小于0就令其等于0,大于2*s就等于2*s   out: 返回
    for x in (labels4[:, 1:], *segments4):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()

    # 测试代码  测试前面的mosaic效果
    # cv2.imshow("mosaic", img4)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(img4.shape)   # (1280, 1280, 3)

    # 随机偏移标签中心，生成新的标签与原标签结合 replicate
    # img4, labels4 = replicate(img4, labels4)
    #
    # # 测试代码  测试replicate效果
    # cv2.imshow("replicate", img4)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(img4.shape)   # (1280, 1280, 3)

    # Augment
    # random_perspective Augment  随机透视变换 [1280, 1280, 3] => [640, 640, 3]
    # 对mosaic整合后的图片进行随机旋转、平移、缩放、裁剪，透视变换，并resize为输入大小img_size
    img4, labels4 = random_perspective(img4, labels4, segments4,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove

    # 测试代码 测试mosaic + random_perspective随机仿射变换效果
    # cv2.imshow("random_perspective", img4)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(img4.shape)   # (640, 640, 3)

    return img4, labels4


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a

def merge_bboxes(bboxes, cutx, cuty):
    merge_bbox = []
    for i in range(len(bboxes)):
        for box in bboxes[i]:
            tmp_box = []
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            if i == 0:
                if y1 > cuty or x1 > cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2 - y1 < 5:
                        continue
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2 - x1 < 5:
                        continue
            if i == 1:
                if y2 < cuty or x1 > cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2 - y1 < 5:
                        continue
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2 - x1 < 5:
                        continue
            if i == 2:
                if y2 < cuty or x2 < cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2 - y1 < 5:
                        continue
                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2 - x1 < 5:
                        continue
            if i == 3:
                if y1 > cuty or x2 < cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2 - y1 < 5:
                        continue
                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2 - x1 < 5:
                        continue
            tmp_box.append(x1)
            tmp_box.append(y1)
            tmp_box.append(x2)
            tmp_box.append(y2)
            tmp_box.append(box[-1])
            merge_bbox.append(tmp_box)
    return merge_bbox

def get_random_data(annotation_line, input_shape, random=True, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    h, w = input_shape
    min_offset_x = 0.4
    min_offset_y = 0.4
    scale_low = 1 - min(min_offset_x, min_offset_y)
    scale_high = scale_low + 0.2
    image_datas = []
    box_datas = []
    index = 0
    place_x = [0, 0, int(w * min_offset_x), int(w * min_offset_x)]
    place_y = [0, int(h * min_offset_y), int(w * min_offset_y), 0]
    for line in annotation_line:
        # 每一行进行分割
        line_content = line.split()
        # 打开图片
        image = Image.open(line_content[0])
        image = image.convert("RGB")
        # 图片的大小
        iw, ih = image.size
        # 保存框的位置
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line_content[1:]])

        # image.save(str(index)+".jpg")
        # 是否翻转图片
        flip = rand() < .5
        if flip and len(box) > 0:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            box[:, [0, 2]] = iw - box[:, [2, 0]]

        # 对输入进来的图片进行缩放
        new_ar = w / h
        scale = rand(scale_low, scale_high)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # 进行色域变换
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = rgb_to_hsv(np.array(image) / 255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x > 1] = 1
        x[x < 0] = 0
        image = hsv_to_rgb(x)

        image = Image.fromarray((image * 255).astype(np.uint8))
        # 将图片进行放置，分别对应四张分割图片的位置
        dx = place_x[index]
        dy = place_y[index]
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image) / 255
        # Image.fromarray((image_data*255).astype(np.uint8)).save(str(index)+"distort.jpg")
        index = index + 1
        box_data = []
        # 对box进行重新处理
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]
            box_data = np.zeros((len(box), 5))
            box_data[:len(box)] = box

        image_datas.append(image_data)
        box_datas.append(box_data)

        img = Image.fromarray((image_data * 255).astype(np.uint8))
        for j in range(len(box_data)):
            thickness = 3
            left, top, right, bottom = box_data[j][0:4]
            draw = ImageDraw.Draw(img)
            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=(255, 255, 255))
        img.show()

    # 将图片分割，放在一起
    cutx = np.random.randint(int(w * min_offset_x), int(w * (1 - min_offset_x)))
    cuty = np.random.randint(int(h * min_offset_y), int(h * (1 - min_offset_y)))

    new_image = np.zeros([h, w, 3])
    new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
    new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
    new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
    new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

    # 对框进行进一步的处理
    new_boxes = merge_bboxes(box_datas, cutx, cuty)
    return new_image, new_boxes

def normal_(annotation_line, input_shape):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    image = Image.open(line[0])
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

    iw, ih = image.size
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    box[:, [0, 2]] = iw - box[:, [2, 0]]

    return image, box


if __name__ == "__main__":
    with open("/Users/fantome/Library/CloudStorage/OneDrive-个人/Git/1_Github/Some_Python_Scripts/scripts/code0037/test_code/mosaic_train.txt") as f:
        lines = f.readlines()
    a = np.random.randint(0, len(lines))
    line = lines[0:4]
    image_data, box_data = get_random_data(line, [416, 416])
    print(type(image_data))
    # img = Image.fromarray((image_data * 255).astype(np.uint8))
    # for j in range(len(box_data)):
    #     thickness = 3
    #     left, top, right, bottom = box_data[j][0:4]
    #     draw = ImageDraw.Draw(img)
    #     for i in range(thickness):
    #         draw.rectangle([left + i, top + i, right - i, bottom - i], outline=(255, 255, 255))
    # img.show()
    # img.save("box_all.jpg")