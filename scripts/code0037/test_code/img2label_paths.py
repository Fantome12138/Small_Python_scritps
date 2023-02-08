def img2label_paths(img_paths):
    """用在LoadImagesAndLabels模块的__init__函数中
    根据imgs图片的路径找到对应labels的路径
    Define label paths as a function of image paths
    :params img_paths: {list: 50}  整个数据集的图片相对路径  例如: '..\\datasets\\VOC\\images\\train2007\\000012.jpg'
                                                        =>   '..\\datasets\\VOC\\labels\\train2007\\000012.jpg'
    """
    # 因为python是跨平台的,在Windows上,文件的路径分隔符是'\',在Linux上是'/'
    # 为了让代码在不同的平台上都能运行，那么路径应该写'\'还是'/'呢？ os.sep根据你所处的平台, 自动采用相应的分隔符号
    # sa: '\\images\\'    sb: '\\labels\\'
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    # 把img_paths中所以图片路径中的images替换为labels
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]