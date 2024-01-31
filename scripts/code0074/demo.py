'''
读取input.md文件，将包含英文字符前后都打上空格，美观
'''

# import re
# def insert_space(text):
#     pattern1 = re.compile('([\u4e00-\u9fa5])([A-Za-z])')  # 中文后面是英文
#     pattern2 = re.compile('([A-Za-z])([\u4e00-\u9fa5])')  # 英文后面是中文
#     text = re.sub(pattern1, r'\1 \2', text)
#     text = re.sub(pattern2, r'\1 \2', text)
#     return text

# # 读取文件内容
# with open('/home/input.md', 'r', encoding='utf-8') as f:
#     text = f.read()

# # 对内容进行处理
# processed_text = insert_space(text)

# # 把处理后的内容写回到文件中
# with open('/home/input.md', 'w', encoding='utf-8') as f:
#     f.write(processed_text)

# -----------------------------------------------------------------------------
'''
replace  photo/ 转 ../photo/
'''

# Open the file in read mode and read its contents
with open('/home/input.md', 'r') as file:
    filedata = file.read()

# Replace the target string
filedata = filedata.replace('<img src="photo/', '<img src="../photo/')

# Open the file in write mode and overwrite it with the new content
with open('/home/input.md', 'w') as file:
    file.write(filedata)