# 1
f = open("foo.txt", encoding='gbk')             # 返回一个文件对象
line = f.readline()             # 调用文件的 readline()方法
while line:              
    print(line, end = '')       # 后面跟' '将忽略换行符
    line = f.readline()
f.close()

# 2
for line in open("foo.txt"):
    print(line, end = '')

# 3
f = open("c:\\1.txt","r")
lines = f.readlines()           # 读取全部内容
for line in lines:
    print(line, end = '')
    txt.append(line.strip())    # 去除首尾空格



