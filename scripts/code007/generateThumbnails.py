import cv2
import os

def generateThumbnails(path):   
    dirs = os.listdir(path)
    thumbnailsPath = path + "/Thumbnails"

    for dir in dirs:
        if os.path.splitext(dir)[1]== ".jpg":
            img = cv2.imread(path+"/"+dir)

            height = img.shape[0]  # 图片高度
            width = img.shape[1]   # 图片宽度
            img1 = cv2.resize(img,(100,int(100/width*height)))    
            if(not os.path.exists(thumbnailsPath)):
                os.makedirs(thumbnailsPath)
            cv2.imwrite(thumbnailsPath+"/"+dir,img1)

if __name__ == '__main__':
    path = "./source"   
    generateThumbnails(path)