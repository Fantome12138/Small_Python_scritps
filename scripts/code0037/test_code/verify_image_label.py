# 这个函数用于检查每一张图片和每一张label文件是否完好。
# 图片文件: 检查内容、格式、大小、完整性
# label文件: 检查每个gt必须是矩形(每行都得是5个数 class+xywh) + 标签是否全部>=0 + 标签坐标xywh是否归一化 + 标签中是否有重复的坐标


def verify_image_label(args):
    """用在cache_labels函数中
    检测数据集中每张图片和每张laebl是否完好
    图片文件: 内容、格式、大小、完整性
    label文件: 每个gt必须是矩形(每行都得是5个数 class+xywh) + 标签是否全部>=0 + 标签坐标xywh是否归一化 + 标签中是否有重复的坐标
    :params im_file: 数据集中一张图片的path相对路径
    :params lb_file: 数据集中一张图片的label相对路径
    :params prefix: 日志头部信息(彩打高亮部分)
    :return im_file: 当前这张图片的path相对路径
    :return l: [gt_num, cls+xywh(normalized)]
               如果这张图片没有一个segment多边形标签 l就存储原label(全部是正常矩形标签)
               如果这张图片有一个segment多边形标签  l就存储经过segments2boxes处理好的标签(正常矩形标签不处理 多边形标签转化为矩形标签)
    :return shape: 当前这张图片的形状 shape
    :return segments: 如果这张图片没有一个segment多边形标签 存储None
                      如果这张图片有一个segment多边形标签 就把这张图片的所有label存储到segments中(若干个正常gt 若干个多边形标签) [gt_num, xy1...]
    :return nm: number missing 当前这张图片的label是否丢失         丢失=1    存在=0
    :return nf: number found 当前这张图片的label是否存在           存在=1    丢失=0
    :return ne: number empty 当前这张图片的label是否是空的         空的=1    没空=0
    :return nc: number corrupt 当前这张图片的label文件是否是破损的  破损的=1  没破损=0
    :return msg: 返回的msg信息  label文件完好=‘’  label文件破损=warning信息
    """
    im_file, lb_file, prefix = args
    nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, corrupt label
    try:
        # 检查这张图片(内容、格式、大小、完整性) verify images
        im = Image.open(im_file)  # 打开图片文件
        im.verify()  # PIL verify 检查图片内容和格式是否正常
        shape = exif_size(im)  # 当前图片的大小 image size
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'  # 图片大小必须大于9个pixels
        assert im.format.lower() in img_formats, f'invalid image format {im.format}'  # 图片格式必须在img_format中
        if im.format.lower() in ('jpg', 'jpeg'):  # 检查jpg格式文件
            with open(im_file, 'rb') as f:
                # f.seek: -2 偏移量 向文件头方向中移动的字节数   2 相对位置 从文件尾开始偏移
                f.seek(-2, 2)
                # f.read(): 读取图片文件  指令: \xff\xd9  检测整张图片是否完整  如果不完整就返回corrupted JPEG
                assert f.read() == b'\xff\xd9', 'corrupted JPEG'

        # verify labels
        segments = []  # 存放这张图所有gt框的信息(包含segments多边形: label某一列数大于8)
        if os.path.isfile(lb_file):  # 如果这个label路径存在
            nf = 1  # label found
            with open(lb_file, 'r') as f:  # 读取label文件
                # 读取当前label文件的每一行: 每一行都是当前图片的一个gt
                l = [x.split() for x in f.read().strip().splitlines() if len(x)]
                # any() 函数用于判断给定的可迭代参数 是否全部为False,则返回 False; 如果有一个为 True,则返回True
                # 如果当前图片的label文件某一列数大于8, 则认为label是存在segment的polygon点(多边形)  就不是矩阵 则将label信息存入segment中
                if any([len(x) > 8 for x in l]):  # is segment
                    # 当前图片中所有gt框的类别
                    classes = np.array([x[0] for x in l], dtype=np.float32)
                    # 获得这张图中所有gt框的label信息(包含segment多边形标签)
                    # 因为segment标签可以是不同长度，所以这里segments是一个列表 [gt_num, xy1...(normalized)]
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]
                    # 获得这张图中所有gt框的label信息(不包含segment多边形标签)
                    # segments(多边形) -> bbox(正方形), 得到新标签  [gt_num, cls+xywh(normalized)]
                    l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)
                l = np.array(l, dtype=np.float32)  # l: to float32
            if len(l):
                # 判断标签是否有五列
                assert l.shape[1] == 5, 'labels require 5 columns each'
                # 判断标签是否全部>=0
                assert (l >= 0).all(), 'negative labels'
                # 判断标签坐标x y w h是否归一化
                assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                # 判断标签中是否有重复的坐标
                assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
            else:
                ne = 1  # label empty  l.shape[0] == 0则为空的标签，ne=1
                l = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing  不存在标签文件，则nm = 1
            l = np.zeros((0, 5), dtype=np.float32)
        return im_file, l, shape, segments, nm, nf, ne, nc, ''
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING: Ignoring corrupted image and/or label {im_file}: {e}'
        return [None, None, None, None, nm, nf, ne, nc, msg]
