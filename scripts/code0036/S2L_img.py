import cv2
import numpy as np

def img_resize_to_target_white(image):
    target = np.ones((224,224),dtype=np.uint8)*255

    ret = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)

    h = image.shape[0]
    w = image.shape[1]
    for i in range(224):
        for j in range(224):
            if(i < h) and (j < w):

                ret[i, j, 0] = image[i, j, 0]
                ret[i, j, 1] = image[i, j, 1]
                ret[i, j, 2] = image[i, j, 2]
            else:
                ret[i, j, 0] = 255
                ret[i, j, 1] = 255
                ret[i, j, 2] = 255

    return ret
   
if __name__ == '__main__':
    image = cv2.imread('/home/yasin/桌面/test.png')
    img_new_white = img_resize_to_target_white(image)
    cv2.imshow("img_new_white", img_new_white)
    cv2.waitKey() 
