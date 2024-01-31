import json  
  
def merge_annotations(file1, file2):  
    # 读取两个 JSON 文件  
    with open(file1, 'r') as f1:  
        data1 = json.load(f1)  
  
    with open(file2, 'r') as f2:  
        data2 = json.load(f2)  
  
    # 提取 "annotations" 字段  
    annotations1 = data1.get('annotations', [])  
    img1 = data1.get('images', [])  
    print(len(annotations1), len(img1))
    
    annotations2 = data2.get('annotations', [])  
    img2 = data2.get('images', [])  
    print(len(annotations2), len(img2))

    # 拼接两个 "annotations" 列表  
    merged_images = img1 + img2
    new_images = []
    for d_i in merged_images:
        if not any(d_i["file_name"] == x["file_name"] for x in new_images):  
            new_images.append(d_i)  
    print(len(new_images))
    
    merged_annotations = annotations1 + annotations2
    new_annotations = []
    for d_a in merged_annotations:
        # print(d_a)
        if not any(d_a["bbox"] == x["bbox"] for x in new_annotations):  
            new_annotations.append(d_a)  
    print(len(new_annotations))
    
    # 将拼接后的列表存储回其中一个 JSON 数据中
    data2['annotations'] = []
    data2['images'] = []
    data2['annotations'] = new_annotations  
    data2['images'] = new_images
  
    # 将结果写入新的 JSON 文件  
    with open(file2, 'w') as f_merged:  
        json.dump(data2, f_merged)


# 使用示例：  
_type = 'val'
file1 = f'/home/LVM_DATA/instances_{_type}.json'
file2 = f'/home/LVM_DATA/instances_{_type}.json'
# merge_annotations(file1, file2)

with open(file2, 'r') as f:  
    data = json.load(f)
    annotations2 = data.get('annotations', [])
    img2 = data.get('images', [])  
    print(len(annotations2), len(img2))


