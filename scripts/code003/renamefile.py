'''该脚本会重新命名，删除原名称！'''
import os
def rename():
    i = 200
    path = 'C:\\Users\\10619\\OneDrive\\saveimg\\'

    filelist = os.listdir(path)   #该文件夹下所有的文件（包括文件夹）
    for files in filelist:   #遍历所有文件
        i = i + 1
        Olddir = os.path.join(path, files)    #原来的文件路径
        if os.path.isdir(Olddir):       #如果是文件夹则跳过
                continue
        filename = 'fake_data'     #文件名
        filetype = '.jpg'        #文件扩展名
        Newdir = os.path.join(path, filename + str(i) + filetype)   #新的文件路径
        os.rename(Olddir, Newdir)    #重命名
    return True

if __name__ == '__main__':
    rename()
    print('Done')


'''该脚本在原名称基础上 newname + oldname'''
import os
path = '/home/tmp/label/'
# 获取该目录下所有文件，存入列表中
f = os.listdir(path)
print(len(f))
print(f[0])
n = 0
i = 0
for i in f:
    # 设置旧文件名（就是路径+文件名）
    oldname = f[n]

    # 设置新文件名
    # newname = 'smoke' + str(n+1)
    newname = 'smoke'
    # 用os模块中的rename方法对文件改名
    os.rename(path+oldname, path+newname+oldname)
    print(oldname, '======>', path+newname+oldname)
    n += 1
# >>> 001291.xml ===> /home/tmp/label/smoke001291.xml