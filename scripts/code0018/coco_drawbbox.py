import cv2
import argparse
import json
import os
from tqdm import tqdm
 
# ground-truth xywh
def gt_select_xywh(args):
    json_file = open(args.gt_json_path)
    infos = json.load(json_file)
    # import pdb;pdb.set_trace()
    images = infos["images"]
    annos = infos["annotations"]
    assert len(images) == len(images)
    for i in tqdm(range(len(images))):
        im_id = images[i]["id"]
        im_path = os.path.join(args.image_path, images[i]["file_name"])
        img = cv2.imread(im_path)
        for j in range(len(annos)):
            category_id = annos[j]["category_id"]
            if annos[j]["image_id"] == im_id:      
                x, y, w, h = annos[j]["bbox"]
                x, y, w, h = int(x), int(y), int(w), int(h)
                x2, y2 = x + w, y + h
                img = cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 255), thickness=2)  # yellow
                img = cv2.putText(img, "{}".format(category_id),(x + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,127,0), 2)
                img_name = os.path.join(args.outpath, images[i]["file_name"].split('/')[-1])
                # import pdb;pdb.set_trace()
                cv2.imwrite(img_name, img)
                # continue
        # print(i)

# ground-truth xyxy
def gt_select_xyxy(args):
    json_file = open(args.gt_json_path)
    infos = json.load(json_file)
    images = infos["images"]
    annos = infos["annotations"]
    assert len(images) == len(images)
    for i in tqdm(range(len(images))):
        im_id = images[i]["id"]
        im_path = os.path.join(args.image_path, images[i]["file_name"])
        img = cv2.imread(im_path)
        for j in range(len(annos)):
            category_id = annos[j]["category_id"]
            if annos[j]["image_id"] == im_id:
                x1, y1, x2, y2 = annos[j]["bbox"]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # object_name = annos[j][""]
                img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), thickness=2)  # yellow
                img = cv2.putText(img, "{}".format(category_id),(x1 + 5, y1 + 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,127,0), 2)
                img_name = os.path.join(args.outpath, images[i]["file_name"].split('/')[-1])
                # import pdb;pdb.set_trace()
                cv2.imwrite(img_name, img)
                # continue
        # print(i)

category_name = {0: 'person', 1:'child'}
# predict
def predict_select(args):
    infos = json.load(open(args.pred_json_path))
    pred_gt_file = json.load(open(args.gt_json_path))
    image_name_id_map = {i['id']: i['file_name'] for i in pred_gt_file['images']}
    # import pdb;pdb.set_trace()
    for i in tqdm(infos):
        im_path = os.path.join(args.image_path, image_name_id_map[i["image_id"]])
        img_name = os.path.join(args.outpath, image_name_id_map[i["image_id"]])
        score = str(i["score"])
        category = i['category_id']
        if not os.path.exists(img_name):
            img = cv2.imread(im_path)
        else: 
            img = cv2.imread(img_name)
        x, y, w, h = i["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)
        x2, y2 = x + w, y + h
        if float(score) >= 0.25:
            cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 255), thickness=2) # red
            cv2.putText(img, "{} {}".format(score, category),(x2, y2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,127,0), 2)
        # else:
        #     cv2.rectangle(img, (x, y), (x2, y2), (255, 255, 0), 2)  # green
        img_name = os.path.join(args.outpath, image_name_id_map[i["image_id"]])
        cv2.imwrite(img_name, img)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start convert.')
    parser.add_argument('--pred_json_path', type=str, help='predition json file path')
    parser.add_argument('--gt_json_path', type=str, help='predition json file path')
    parser.add_argument('--image_path', type=str, help='raw image data dir')
    parser.add_argument('--outpath', type=str, help='image box dir')
    args = parser.parse_args()
    # predict_select(args) # predict json draw box
    gt_select_xywh(args) # gt json draw box (xywh)
    # gt_select_xyxy(args) # gt json draw box (xyxy)
