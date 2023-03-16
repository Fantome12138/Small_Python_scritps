import os
import random
 
num = 0;
for image_name in os.listdir("/home/cabinet_data/image_tmp2/"):
    feed = random.randint(0, 10)
    if feed <= 5:
        os.remove(os.path.join("/home/cabinet_data/image_tmp2/", image_name));
        print(feed)
        print(os.path.join("/home/cabinet_data/image_tmp2/", image_name))
        num = num + 1
print(num)