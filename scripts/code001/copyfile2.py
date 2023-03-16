import os
import shutil

def mycopyfile(srcfile, dstpath):                       # 复制函数
    if not os.path.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        fpath, fname = os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        shutil.copy(srcfile, dstpath + fname)          # 复制文件
        print ("copy %s -> %s"%(srcfile, dstpath + fname))

def get_filename(dir):
    fs = [] 
    for root, dirs, files in os.walk(dir):
        for name in files:
            _, ending = os.path.splitext(name)
            if ending == '.jpg':
                fs.append(os.path.join(root,name))
    return fs

dir = '/home/OccludedQR150/1/hik_img/'
fs = get_filename(dir)
print(fs)
dstpath = '/home/cabinet_data/image/'
for item in fs:
    mycopyfile(item, dstpath)
    pass

