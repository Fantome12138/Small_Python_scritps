import os
# 定义几个路径
path1 = "/home/LVM_DATA/train_images"
path2 = "/home/LVM_DATA/train_labels"
path3 = "/home/LVM_DATA/val_images"
path4 = "/home/LVM_DATA/val_labels"
# 将路径放入一个列表中
paths = [path1, path2, path3, path4]
# 遍历每个路径
for path in paths:
    # 初始化一个计数器
    count = 0
    # 获取路径下的文件和子目录的列表
    files = os.listdir(path)
    # 遍历列表中的每个元素
    for file in files:
        # 拼接成完整的路径
        file_path = os.path.join(path, file)
        # 判断是否为文件
        if os.path.isfile(file_path):
            # 累加文件数
            count += 1
    # 打印路径和文件数
    print(f"{path} has {count} files")
