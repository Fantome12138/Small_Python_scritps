import os
import glob

# 定义源目录和目标目录
src_dir = '/Users/fantome/Downloads/新浦化学/2023-10-16-新采样日间样本/images/'
dst_dir = '/Users/fantome/Downloads/新浦化学/2023-10-16-新采样日间样本/labels/'

count = 0
for filename in glob.glob(os.path.join(src_dir, '*.jpg')):
    count += 1
    # 构造新的文件名，即原文件名加上"led"
    jpg_filename = os.path.splitext(filename)[0]
    new_filename = 'new1_day' + str(count)
    new_jpg_name = new_filename + '.jpg'
    # 使用os.rename函数将原文件名修改为新的文件名
    os.rename(filename, new_jpg_name)
    # 在目标目录中查找是否有与新的文件名相同.txt文件
    for txt_filename in glob.glob(os.path.join(dst_dir, '*.txt')):
        if os.path.splitext(txt_filename)[0] == jpg_filename:
            # 如果找到了匹配的.txt文件，就修改其文件名
            new_txt_name = new_jpg_name + '.txt'
            os.rename(txt_filename, new_txt_name)
            break
