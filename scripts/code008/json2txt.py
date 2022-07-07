import os
import json

json_dir = 'train_annos.json'  # json文件路径
out_dir = 'output/'  # 输出的 txt 文件路径

def main():
    # 读取 json 文件数据
    with open(json_dir, 'r') as load_f:
        content = json.load(load_f)
    # 循环处理
    for t in content:
        tmp = t['name'].split('.')
        filename = out_dir + tmp[0] + '.txt'

        if os.path.exists(filename):
            # 计算 yolo 数据格式所需要的中心点的 相对 x, y 坐标, w,h 的值
            x = (t['bbox'][0] + t['bbox'][2]) / 2 / t['image_width']
            y = (t['bbox'][1] + t['bbox'][3]) / 2 / t['image_height']
            w = (t['bbox'][2] - t['bbox'][0]) / t['image_width']
            h = (t['bbox'][3] - t['bbox'][1]) / t['image_height']
            fp = open(filename, mode="r+", encoding="utf-8")
            file_str = str(t['category']) + ' ' + str(round(x, 6)) + ' ' + str(round(y, 6)) + ' ' + str(round(w, 6)) + \
                       ' ' + str(round(h, 6))
            line_data = fp.readlines()

            if len(line_data) != 0:
                fp.write('\n' + file_str)
            else:
                fp.write(file_str)
            fp.close()

        # 不存在则创建文件
        else:
            fp = open(filename, mode="w", encoding="utf-8")
            fp.close()


if __name__ == '__main__':
    main()

    
#######################################
# -*- coding: utf-8 -*-
#!/usr/bin/python3
import json
import os

name2id = {'pressure_meter': 0, 'level_meter':1, 'ammeter':2}

def convert(img_size, box):
    dw = 1. / (img_size[0])
    dh = 1. / (img_size[1])
    x = (box[0] + box[2]) / 2.0 - 1
    y = (box[1] + box[3]) / 2.0 - 1
    w = abs(box[2] - box[0])
    h = abs(box[3] - box[1])
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def decode_json(json_floder_path, json_name):
    txt_name = 'C:\\Users\\label\\' + json_name[0:-5] + '.txt'
    txt_file = open(txt_name, 'w')

    json_path = os.path.join(json_floder_path, json_name)
    data = json.load(open(json_path, 'r', encoding="ISO-8859-1"))

    img_w = data['imageWidth']
    img_h = data['imageHeight']

    for i in data['shapes']:

        label_name = i['label']
        if (i['shape_type'] == 'rectangle'):
            x1 = int(i['points'][0][0])
            y1 = int(i['points'][0][1])
            x2 = int(i['points'][1][0])
            y2 = int(i['points'][1][1])

            bb = (x1, y1, x2, y2)
            bbox = convert((img_w, img_h), bb)
            txt_file.write(str(name2id[label_name]) + " " + " ".join([str(a) for a in bbox]) + '\n')


if __name__ == "__main__":

    json_floder_path = 'C:\\Users\\Downloads\\json\\'
    json_names = os.listdir(json_floder_path)
    for json_name in json_names:
        decode_json(json_floder_path, json_name)
