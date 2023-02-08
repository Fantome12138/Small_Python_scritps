import os
import glob
def get_latest_run(search_dir='.'):
    """
    用在train.py查找最近的pt文件进行断点续训
    用于返回该项目中最近的模型 'last.pt'对应的路径
    
    :params search_dir: 要搜索的文件的根目录 默认是 '.'  表示搜索该项目中的文件
    """
    # 从Python版本3.5开始, glob模块支持该"**"指令（仅当传递recursive标志时才会解析该指令)
    # glob.glob函数匹配所有的符合条件的文件, 并将其以list的形式返回
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    print(last_list)
    # os.path.getctime 返回路径对应文件的创建时间
    # 所以这里是返回路径列表中创建时间最晚(最近的last文件)的路径
    return max(last_list, key=os.path.getctime) if last_list else ''

ckpt = get_latest_run(search_dir="./")
print(ckpt)
assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'


from pathlib import Path 
def check_file(file):
    """
    用在train.py、detect.py、test.py等文件中检查本地有没有这个文件
    检查相关文件路径能否找到文件 并返回文件名
    Search/download file (if necessary) and return path
    """
    file = str(file)  # convert to str()
    # 如果传进来的是文件或者是’‘, 直接返回文件名str
    if Path(file).is_file() or file == '':  # exists
        return file  
    else:
        # 否则, 传进来的就是当前项目下的一个全局路径 查找匹配的文件名 返回第一个
        # glob.glob: 匹配当前项目下的所有项目 返回所有符合条件的文件files
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), f'File not found: {file}'  # assert file was found
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
        # 返回第一个匹配到的文件名
        return files[0]  # return file
a = check_file('./test.py')
print(a)
print(a.endswith('.py'))



import re
def increment_path(path, exist_ok=False, sep='', mkdir=False):
    """这是个用处特别广泛的函数 train.py、detect.py、test.py等都会用到
    递增路径 如 run/train/exp --> runs/train/exp{sep}0, runs/exp{sep}1 etc.
    :params path: window path   run/train/exp
    :params exist_ok: False
    :params sep: exp文件名的后缀  默认''
    :params mkdir: 是否在这里创建dir  False
    """
    path = Path(path)  # string/win路径 -> win路径
    # 如果该文件夹已经存在 则将路径run/train/exp修改为 runs/train/exp1
    if path.exists() and not exist_ok:
        # path.suffix 得到路径path的后缀  ''
        suffix = path.suffix
        # .with_suffix 将路径添加一个后缀 ''
        path = path.with_suffix('')
        # 模糊搜索和path\sep相似的路径, 存在一个list列表中 如['runs\\train\\exp', 'runs\\train\\exp1']
        # f开头表示在字符串内支持大括号内的python表达式
        dirs = glob.glob(f"{path}{sep}*")
        # r的作用是去除转义字符       path.stem: 没有后缀的文件名 exp
        # re 模糊查询模块  re.search: 查找dir中有字符串'exp/数字'的d   \d匹配数字
        # matches [None, <re.Match object; span=(11, 15), match='exp1'>]  可以看到返回span(匹配的位置) match(匹配的对象)
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        # i = [1]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        # 生成需要生成文件的exp后面的数字 n = max(i) + 1 = 2
        n = max(i) + 1 if i else 1  # increment number
        # 返回path runs/train/exp2
        path = Path(f"{path}{sep}{n}{suffix}")  # update path

    # path.suffix文件后缀   path.parent　路径的上级目录  runs/train/exp2
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:  # mkdir 默认False 先不创建dir
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path  # 返回runs/train/exp2


'''得到path路径下的所有图片的路径img_files'''
prefix = ''
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
path = '/home/YangziData/'
f = []  # image files
for p in path if isinstance(path, list) else [path]:
    # 获取数据集路径path，包含图片路径的txt文件或者包含图片的文件夹路径
    # 使用pathlib.Path生成与操作系统无关的路径，因为不同操作系统路径的‘/’会有所不同
    p = Path(p)  # os-agnostic
    # 如果路径path为包含图片的文件夹路径
    if p.is_dir():  # dir
        # glob.glab: 返回所有匹配的文件路径列表  递归获取p路径下所有文件
        f += glob.glob(str(p / '**' / '*.*'), recursive=True)
        # f = list(p.rglob('**/*.*'))  # pathlib
    # 如果路径path为包含图片路径的txt文件
    elif p.is_file():  # file
        with open(p, 'r') as t:
            t = t.read().strip().splitlines()  # 获取图片路径，更换相对路径
            # 获取数据集路径的上级父目录  os.sep为路径里的分隔符（不同路径的分隔符不同，os.sep可以根据系统自适应）
            parent = str(p.parent) + os.sep
            f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
            # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
    else:
        raise Exception(f'{prefix}{p} does not exist')
# 破折号替换为os.sep，os.path.splitext(x)将文件名与扩展名分开并返回一个列表
# 筛选f中所有的图片文件
img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in img_formats])
# self.img_files = sorted([x for x in f if x.suffix[1:].lower() in img_formats])  # pathlib
assert img_files, f'{prefix}No images found'
print(img_files, '\nnum of img_files:', len(img_files), type(img_files))

sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
# 把img_paths中所以图片路径中的images替换为labels
label_files = [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_files]
print(len(label_files), type(label_files))

cache_path = (p if p.is_file() else Path(label_files[0]).parent).with_suffix('.cache')
print(type(cache_path))  # <class 'pathlib.PosixPath'>


import math
def make_divisible(x, divisor):
    """用在下面的make_divisible函数中  yolo.py的parse_model函数和commom.py的AutoShape函数中
    取大于等于x且是divisor的最小倍数
    Returns x evenly divisible by divisor
    """
    # math.ceil 向上取整
    return math.ceil(x / divisor) * divisor

result1 = make_divisible(100, 200)
print(result1)

def check_img_size(img_size, s=32):
    """这个函数主要用于train.py中和detect.py中  用来检查图片的长宽是否符合规定
    检查img_size是否能被s整除，这里默认s为32  返回大于等于img_size且是s的最小倍数
    Verify img_size is a multiple of stride s
    """
    # 取大于等于x的最小值且该值能被divisor整除
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size

new_size = check_img_size(1280, 32)
print(new_size)


def clean_str(s):
    """在datasets.py中的LoadStreams类中被调用
    字符串s里在pattern中字符替换为下划线_  注意pattern中[]不能省
    Cleans a string by replacing special characters with underscore _
    """
    # re: 用来匹配字符串（动态、模糊）的模块  正则表达式模块
    # pattern: 表示正则中的模式字符串  repl: 就是replacement的字符串  string: 要被处理, 要被替换的那个string字符串
    # 所以这句话执行的是将字符串s里在pattern中的字符串替换为 "_"
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)
str1 = '12341@#$sdfdsa'
str1 = clean_str(str1)
print(str1)


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    """这是个用处特别广泛的函数 train.py、detect.py、test.py等都会用到
    递增路径 如 run/train/exp --> runs/train/exp{sep}0, runs/exp{sep}1 etc.
    :params path: window path   run/train/exp
    :params exist_ok: False
    :params sep: exp文件名的后缀  默认''
    :params mkdir: 是否在这里创建dir  False
    """
    path = Path(path)  # string/win路径 -> win路径
    # 如果该文件夹已经存在 则将路径run/train/exp修改为 runs/train/exp1
    if path.exists() and not exist_ok:
        # path.suffix 得到路径path的后缀  ''
        suffix = path.suffix
        # .with_suffix 将路径添加一个后缀 ''
        path = path.with_suffix('')
        # 模糊搜索和path\sep相似的路径, 存在一个list列表中 如['runs\\train\\exp', 'runs\\train\\exp1']
        # f开头表示在字符串内支持大括号内的python表达式
        dirs = glob.glob(f"{path}{sep}*")
        # r的作用是去除转义字符       path.stem: 没有后缀的文件名 exp
        # re 模糊查询模块  re.search: 查找dir中有字符串'exp/数字'的d   \d匹配数字
        # matches [None, <re.Match object; span=(11, 15), match='exp1'>]  可以看到返回span(匹配的位置) match(匹配的对象)
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        # i = [1]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        # 生成需要生成文件的exp后面的数字 n = max(i) + 1 = 2
        n = max(i) + 1 if i else 1  # increment number
        # 返回path runs/train/exp2
        path = Path(f"{path}{sep}{n}{suffix}")  # update path

    # path.suffix文件后缀   path.parent　路径的上级目录  runs/train/exp2
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:  # mkdir 默认False 先不创建dir
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path  # 返回runs/train/exp2


