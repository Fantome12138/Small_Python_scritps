# 当然，无论是YOLO还是opencv，都是老外开发的，开发的过程中肯定不会考虑中文显示了，所以一直以来，在opencv-python中显示中文都有一些麻烦。那如何才能在YOLOv5图像识别中让标签变为中文呢？这里提供了一种修改YOLOv5源码方法可以参考。

# YOLOv5的使用在这里就不再阐述了，我们直接在YOLOv5程序中utils/utils.py（新版的是utils/general.py）文件下找到这一行代码：

def plot_one_box(x, img, color=None, label=None, line_thickness=None):

# 这个函数主要是将我们识别出的目标在图片上框出来并标记上文字标签。opencv-python不能直接显示中文信息，因此我们需要进行以下3个步骤

# 将opencv图片格式转换成PIL的图片格式；
# 使用PIL绘制文字；
# PIL图片格式转换成oepncv的图片格式；
# 将这个函数稍微修改，如下所示：

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=30):
    # 图像从OpenCV格式转换成PIL格式
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("../Font/simhei.ttf", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def plot_one_box(x, img, color=None, label=None, ch_text=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        # cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        img_text = cv2ImgAddText(img, ch_text, c1[0], c2[1], (255, 255, 255), 25)  # todo: 文字以左上点坐标为准，即：c1[0], c2[1]
    return img_text

# 此时，底层代码部分我们已经修改好了，但是到这里还没有结束，我们还需修改detect.py中的代码。找到如下位置：
# Write results
for *xyxy, conf, cls in det:

# 对此处代码进行修改，将识别出的标签用中文显示出来，我这里的标签一共有4类，分别为trash，sewage_well_cover，road_cracks，pavement_pit，对应的中文信息依次为垃圾桶完好，污水井盖完好，道路裂缝，路面坑槽。现在将英文标签用中文显示出来。
# Write results
for *xyxy, conf, cls in det:
    if save_txt:  # Write to file
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
            file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

    if save_img or view_img:  # Add bbox to image
        # ch_text = '%s,长:200cm,面积:2.3m2,%.2f' % ('物体', conf)
        # print(names[int(cls)])
        # 根据像素确定长宽和对角线像素长度
        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
        weight = abs(c2[1]-c1[1])
        length = abs(c2[0]-c1[0])
        diagonal = round(((c2[0]-c1[0])**2 + (c2[1]-c1[1])**2)**0.5, 2)
        areas = round(abs(c2[1]-c1[1]) * abs(c2[0]-c1[0]), 2)
        label = '%s %.2f x=%.2f,y=%.2f' % (names[int(cls)], conf, int(xyxy[0]), int(xyxy[1]))  # 可以不用
        # 设置固定颜色
        color_dict = {'1': [220,20,60], '2': [75,195,185], '3': [255,165,0], '4': [60,20,220]}
        # 中文输出
        if names[int(cls)] == 'trash':
            ch_text = '%s，%.2f' % ('垃圾桶完好', conf)
            color_single = color_dict['1']
        elif names[int(cls)] == 'sewage_well_cover':
            ch_text = '%s，%.2f' % ('污水井盖完好', conf)
            color_single = color_dict['2']
        elif names[int(cls)] == 'road_cracks':
            ch_text = '%s，裂缝长度:%s，%.2f' % ('道路裂缝', diagonal, conf)
            color_single = color_dict['3']
        elif names[int(cls)] == 'pavement_pit':
            ch_text = '%s，坑槽面积:%s，%.2f' % ('路面坑槽', areas, conf)
            color_single = color_dict['4']
        # im0 = plot_one_box(xyxy, im0, label=label, ch_text=ch_text, color=colors[int(cls)], line_thickness=3)
        im0 = plot_one_box(xyxy, im0, label=label, ch_text=ch_text, color=color_single, line_thickness=3)




