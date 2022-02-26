import os    

for i in range(1,6): 
    path1 = '/home/test/creat_folder/' #设置创建后文件夹存放的位置,此处是在creat_folder文件夹下的1文件夹下创建一组txt
    path2 = '测试_' + str(i)                   
    path = os.path.join(path1,path2)           
    f = open( path + '.txt',"a")   
    f.write("")		
    f.close()		