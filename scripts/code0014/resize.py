import os
import cv2
 
# obtain the filename
path_ori = '/home/jpg/'
filename = os.listdir(path_ori)
 
# resize the image
for fn in filename:
    img = cv2.imread(path_ori + fn)
    dim = (1920, 1080)
    img_res = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite(path_ori+fn, img_res)