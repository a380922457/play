# encoding: utf-8
from random import normalvariate
import numpy as np
import os
from PIL import Image
from random import randint


IMAGE_PATH = "/Users/lianghangming/Desktop/"

# 等比例地把图片较短的一边缩放到区间[256,480]
def rescale(image):
    w = image.size[0]
    h = image.size[1]
    sizeMax = 480
    sizeMin = 256
    random_size = randint(sizeMin, sizeMax)
    if w < h:
        return image.resize((random_size, round(h/w * random_size)))
    else:
        return image.resize((round(w/h * random_size), random_size))


# 随机裁剪图片
def random_crop(image):
    w = image.size[0]
    h = image.size[1]
    size = 224
    new_left = randint(0, w - size)
    new_upper = randint(0, h - size)
    return image.crop((new_left, new_upper, size+new_left, size+new_upper))


# 水平翻转图片
def horizontal_flip(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)


# 图片均值归一化
def nomalizing(image, mean_value, add_num):
    image = np.array(image)
    # image = image.astype(float)
    for i in range(3):
        add_num = add_num.astype(int)
        image[:, :, i] = (image[:, :, i] - mean_value[i] + add_num[i]) #/std[i]
    return image


# 求整个训练集图片的均值
def mean(image_dir):
    for file in os.listdir(image_dir):
        if file.endswith("jpg"):
            file = os.path.join(image_dir, file)
            image = np.array(Image.open(file))
            image = np.reshape(image, [-1, 3])
            try:
                image_array = np.concatenate((image_array, image), 0)  # 第一张图片不存在image_array
            except:
                image_array = image
    mean_value = image_array.mean(0)
    # std_value = image_array.std(0)
    return mean_value


# 求整个训练集图片的PCA
def pca(image_dir, mean_value):
    for file in os.listdir(image_dir):
        if file.endswith("jpg"):
            file = os.path.join(image_dir, file)
            image = np.array(Image.open(file))
            image = np.reshape(image, [-1, 3])
            image = image.astype(float)
            image -= mean_value  # 零均值化
            # image = image/255.0
            try:
                image_array = np.concatenate((image_array, image), 0)  # 第一张图片不存在image_array
            except:
                image_array = image

    # 求协方差矩阵
    image_cov = np.cov([image_array[:, 0], image_array[:, 1], image_array[:, 2]])
    lambd, p = np.linalg.eig(image_cov)
    alpha0 = normalvariate(0, 0.1)
    alpha1 = normalvariate(0, 0.1)
    alpha2 = normalvariate(0, 0.1)
    v = np.transpose((alpha0*lambd[0], alpha1*lambd[1], alpha2*lambd[2]))
    add_num = np.dot(v, np.transpose(p))
    return add_num


