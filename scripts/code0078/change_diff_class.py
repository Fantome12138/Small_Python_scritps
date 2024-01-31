import os
 
txt_path = '/home/LVM_DATA/new_untrain/ma_labels/'  # txt文件所在路径

## ALL
class_num = 38  # 样本类别数
labels = ['meter_ocr', 'circle', 'ampermeter', 'r_light_on', 'r_light_off', 'g_light_on',
    'g_light_off', 'y_light_on', 'y_light_off', 'b_light_on', 'b_light_off',
    'o_light_on', 'o_light_off', 'w_light_on', 'w_light_off', 'energy_storage_ind',
    'rotate_switch', 'pannel_quickcut', 'pannel_meactl', 'pannel_protect', 'pannel_wavelim', 
    'pannel_powctl', 'pannel_amprotect', 'pannel_led', 'pannel_voltage','energy_storage_ind_1',
    'energy_storage_ind_2','energy_storage_ind_3','energy_storage_ind_4','rb_light_on', 'rb_light_off', 'gb_light_on',
    'gb_light_off', 'yb_light_on', 'yb_light_off', 'wb_light_on', 'wb_light_off','magnetic_flap']

# class_num = 29  # 样本类别数
# labels = ['meter_ocr', 'circle', 'ampermeter', 'r_light_on', 'r_light_off', 'g_light_on',
# 'g_light_off', 'y_light_on', 'y_light_off', 'b_light_on', 'b_light_off',
# 'o_light_on', 'o_light_off', 'w_light_on', 'w_light_off', 'energy_storage_ind',
# 'rotate_switch', 'pannel_quickcut', 'pannel_meactl', 'pannel_protect', 'pannel_wavelim', 
# 'pannel_powctl', 'pannel_amprotect', 'pannel_led', 'pannel_voltage','energy_storage_ind_1','energy_storage_ind_2','energy_storage_ind_3','energy_storage_ind_4']

# bgd
# class_num = 10
# labels = ['r_light_on', 'r_light_off', 'g_light_on',
#         'g_light_off', 'y_light_on', 'y_light_off',
#         'o_light_on', 'o_light_off', 'w_light_on', 'w_light_off']

# sheng
# class_num = 8
# labels = ['rb_light_on', 'rb_light_off', 'gb_light_on','gb_light_off',
#           'yb_light_on', 'yb_light_off','wb_light_on', 'wb_light_off']

# xinpu
# class_num = 4
# labels = ['rb_light_on', 'rb_light_off', 'gb_light_on','gb_light_off']

## magnetic_flap
# class_num = 1
# labels = ['magnetic_flap']


all_labels = {}
 
class_list = [i for i in range(class_num)]
class_num_list = [0 for i in range(class_num)]
labels_list = os.listdir(txt_path)
for i in labels_list:
    file_path = os.path.join(txt_path, i)
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    for i, line in enumerate(lines):
        columns = line.split()
        # if int(columns[0]) == 0:
        #     columns[0] = '3'
        #     new_line = ' '.join(columns) + '\n'  
        #     lines[i] = new_line  
        #     continue
        # if int(columns[0]) == 1:
        #     columns[0] = '4'
        #     new_line = ' '.join(columns) + '\n'  
        #     lines[i] = new_line  
        #     continue
        # if int(columns[0]) == 2:
        #     columns[0] = '5'
        #     new_line = ' '.join(columns) + '\n'  
        #     lines[i] = new_line  
        #     continue
        # if int(columns[0]) == 3:
        #     columns[0] = '6'
        #     new_line = ' '.join(columns) + '\n'  
        #     lines[i] = new_line  
        #     continue
        # if int(columns[0]) == 4:
        #     columns[0] = '7'
        #     new_line = ' '.join(columns) + '\n'  
        #     lines[i] = new_line  
        #     continue
        # if int(columns[0]) == 5:
        #     columns[0] = '8'
        #     new_line = ' '.join(columns) + '\n'  
        #     lines[i] = new_line  
        #     continue
        # if int(columns[0]) == 6:
        #     columns[0] = '11'
        #     new_line = ' '.join(columns) + '\n'  
        #     lines[i] = new_line  
        #     continue
        # if int(columns[0]) == 7:
        #     columns[0] = '12'
        #     new_line = ' '.join(columns) + '\n'  
        #     lines[i] = new_line  
        #     continue
        # if int(columns[0]) == 8:
        #     columns[0] = '13'
        #     new_line = ' '.join(columns) + '\n'  
        #     lines[i] = new_line  
        #     continue
        # if int(columns[0]) == 9:
        #     columns[0] = '14'
        #     new_line = ' '.join(columns) + '\n'  
        #     lines[i] = new_line  
        #     continue
        
        
        # if int(columns[0]) == 0:
        #     columns[0] = '29'
        #     new_line = ' '.join(columns) + '\n'  
        #     lines[i] = new_line  
        #     continue
        # if int(columns[0]) == 1:
        #     columns[0] = '30'
        #     new_line = ' '.join(columns) + '\n'  
        #     lines[i] = new_line  
        #     continue
        # if int(columns[0]) == 2:
        #     columns[0] = '31'
        #     new_line = ' '.join(columns) + '\n'  
        #     lines[i] = new_line  
        #     continue
        # if int(columns[0]) == 3:
        #     columns[0] = '32'
        #     new_line = ' '.join(columns) + '\n'  
        #     lines[i] = new_line  
        #     continue
        # new_line = ' '.join(columns) + '\n'  
        # lines[i] = new_line  
    with open(file_path, 'w') as file:
        file.writelines(lines)
        class_ind = class_list.index(int(columns[0]))
        class_num_list[class_ind] += 1
        
    
# 输出每一类的数量以及总数
print(class_num_list)
print('total:', sum(class_num_list))

for i, item in enumerate(labels):
    all_labels.update({item:class_num_list[i]})
print(all_labels)
