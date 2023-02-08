from PIL import Image
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import imageio

# skimage
def read_image_5(path):
    """ skimage

    Args:
        path: path of the image.
    """
    image = io.imread(path)
    io.imshow(image)
    new_path = path[:-4] + '-fc5.jpg'
    io.imsave(new_path, image)
    io.show()

# Imageio  Imageio只用于读取和保存，用matplot保存
def read_image_4(path):
    """ imageio

    Args:
        path: path of the image.
    """
    image = imageio.imread(path)
    plt.imshow(image)
    plt.show()
    new_path = path[:-4] + '-fc4.jpg'
    imageio.imsave(new_path, image)

# opencv 读取图片排列方式为bgr 而非rgb
def read_image_3(path):
    """ OpenCV

    Args:
        path: path of the image.
    """
    image = cv2.imread(path)
    cv2.imshow(path, image)
    cv2.waitKey()
    new_path = path[:-4] + '-fc3.jpg'
    cv2.imwrite(new_path, image)

# Matplot
def read_image_2(path):
    """ matplot

    Args:
        path: path of the image.
    """
    image = mpimg.imread(path)
    plt.imshow(image)
    plt.axis('off')
    new_path = path[:-4] + '-fc2.jpg'
    mpimg.imsave(new_path, image)
    plt.show()

# pillow PIL
def read_image_1(path):
    """ Pillow
    This will show picture with default image showing program.

    Args:
        path: path of the image.
    """
    image = Image.open(path)
    image.show()
    new_path = path[:-4] + '-fc1.jpg'
    image.save(new_path)