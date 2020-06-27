import h5py
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
import scipy

orig_picture = r'E:\近期作业\AI_HW\hwtrainset'

gen_picturn = r'E:\近期作业\AI_HW\raw_data'

classes = ["1","2","3","4","5","6","7","8","9","10"]


def get_traindata(orig_dir, gen_dir, classes):
    i = 0
    for index, name in enumerate(classes):
        class_path = orig_dir + '\\' + name + '\\'
        gen_train_path = gen_dir + '\\' + name
        folder = os.path.exists(gen_train_path)
        if not folder:
            os.makedirs(gen_train_path)
            print(gen_train_path, 'new file')
        else:
            print('There is this flie')
        # 给图片加编号保存
        for imagename_dir in os.listdir(class_path):
            i += 1
            origimage_path = class_path + imagename_dir
            image_data = Image.open(origimage_path).convert('RGB')
            image_data = image_data.resize((64, 64))
            image_data.save(gen_train_path + '\\' + name + str(i) + '.jpg')
            num_samples = i
    print('picturn ：%d' % num_samples)


if __name__ == '__main__':
    get_traindata(orig_picture, gen_picturn, classes)
