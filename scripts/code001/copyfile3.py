import os
import shutil

list1 = []
pannel_list = ['17','18','19','20','21','22']
new_path = '/home/Xinte_Data/pannel/'
txt_dir = '/home/Xinte_Data/all_labels/'
count = 0

for root, dirs, files in os.walk(txt_dir):
    for file in files:
        try:
            txt = open(root+file, 'r')   
        except FileNotFoundError:
            print('File is not found')
        else:
            lines = txt.readlines()
            for line in lines:
                a = line.split()
                x = a[0]
                if x in pannel_list:
                    a = os.path.splitext(file)
                    temptargetname = '%s.jpg' % a[0]
                    shutil.copy(os.path.join('/home/Xinte_Data/all_images/', temptargetname), new_path)
                    print('@@@@@@@@', x, temptargetname)
                    count += 1
                    break
                else: continue
            txt.close()
print(count)