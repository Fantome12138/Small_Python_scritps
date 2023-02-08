import os

'''
统计一个文件的有效代码行数
'''
def count_lines(file_name):
	fp = open(file_name, 'r+',encoding = 'utf-8')
	line_count = 0
	for line in fp.readlines():
		if not line.split():	# 判断是否为空行
			line.strip()		# 去除空行
			continue
		else:
			line_count += 1
	fp.close()
	return line_count

'''
获取目录下文件名并记录每个文件的行数
'''
def mulu_statics(file_dir):
	record_file = open("../mulu_statics.txt", "w+")			# 结果保存至该文件，没有文件就新建文件
	for root, dirs, files in os.walk(file_dir):
		for file in files:
			# 不计算本文件
			if file == "file_statics.py":
				continue
			file_type = file.split('.')						# 只统计规范命名文件：[文件名.扩展名]
			if len(file_type) > 1 :
				if file_type[1] not in ["c", "cpp", "h", "py"]:	# 要统计的文件类型
					continue
			else:
				continue
			file_name = root +"/" + file
			lines = count_lines(file_name)
			print(file + '\t' + repr(lines), file = record_file)
	record_file.close()


if __name__ == '__main__':
    # cur_path
    cur_path = os.path.split(os.path.realpath(__file__))[0]
    print(cur_path)
    mulu_statics(cur_path)
    


