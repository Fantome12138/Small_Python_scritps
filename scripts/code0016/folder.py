import os	
                                       
for i in range(1,11):                           
    path1 = '/home/test/creat_folder/'
    path2 = 'whoami' + str(i)
    path = os.path.join(path1, path2) # 路径拼接
    isExist = os.path.exists(path)    # 定义一个变量判断文件是否存在
    if not isExist:		              # 如果文件不存在,则创建文件夹，并提示创建成功
        os.makedirs(path)	
        print("%s 目录创建成功"%i)
    else:                           
        print("%s 目录已经存在"%i)	        
        continue	

'''或'''
import os    

for i in range(21,30): 
    path1 = 'C:\\Users\\10619\\OneDrive\\Git\\Github\\Some_Python_Scripts\\scripts\\' 
    path2 = 'code' + '00' + str(i)                   
    path = os.path.join(path1, path2)           
    os.makedirs(path)