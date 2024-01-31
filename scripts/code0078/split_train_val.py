# import os
# import random

# def split_images(src_dir, train_dir, val_dir, ratio=0.8):
#     # 遍历源文件夹中的所有文件和子文件夹
#     for root, dirs, files in os.walk(src_dir):
#         for file in files:
#             # 检查是否为jpg文件
#             if file.endswith('.jpg') or file.endswith('.png'):
#                 # 获取完整的文件路径
#                 src_file = os.path.join(root, file)
                
#                 # 根据给定的比例决定图片去向
#                 if random.random() < ratio:
#                     dest_file = os.path.join(train_dir, file)
#                 else:
#                     dest_file = os.path.join(val_dir, file)

#                 # 移动文件
#                 os.rename(src_file, dest_file)

# # 使用方法
# _project_name = 'bgd'
# split_images(f'/home/LVM_DATA/{_project_name}/all_images/', 
#              f'/home/LVM_DATA/{_project_name}/train2017/', 
#              f'/home/LVM_DATA/{_project_name}/val2017/', 0.8)



'''
有两个文件夹/home/images,/home/labels分别存放.jpg和.txt文件，
我希望随机选images里20%的文件到指定路径，
同时复制labels同名的txt文件到指定路径，根据此编写python程序
'''
import os
import shutil
import random

# 定义源文件夹和目标文件夹
_type = 'val'
src_folder_images = f"/home/LVM_DATA/new_untrain/images/"
src_folder_labels = f"/home/LVM_DATA/new_untrain/labels/"
dst_folder_images = f"/home/LVM_DATA/new_untrain/{_type}_images/"
dst_folder_labels = f"/home/LVM_DATA/new_untrain/{_type}_labels/"

if not os.path.exists(dst_folder_images):
    os.makedirs(dst_folder_images)
if not os.path.exists(dst_folder_labels):
    os.makedirs(dst_folder_labels)

# 列出源文件夹中的所有文件
files_images = [f for f in os.listdir(src_folder_images) if f.endswith('.jpg')]
files_labels = [f for f in os.listdir(src_folder_labels) if f.endswith('.txt')]

# 随机选择20%的文件
selected_files = random.sample(files_images, int(len(files_images) * 0.2))

# 对于每个选定的文件，找到同名的.txt文件
for file in selected_files:
    base_name = os.path.splitext(file)[0]  # 获取文件名（不包括扩展名）
    corresponding_txt = base_name + '.txt'  # 构造对应的.txt文件名

    # 检查对应的.txt文件是否存在
    if corresponding_txt in files_labels:
        # 将选定的.jpg和.txt文件复制到目标文件夹
        shutil.move(os.path.join(src_folder_images, file), os.path.join(dst_folder_images, file))
        shutil.move(os.path.join(src_folder_labels, corresponding_txt), os.path.join(dst_folder_labels, corresponding_txt))

