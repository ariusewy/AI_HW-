import os
import numpy as np
from PIL import Image
import h5py
import scipy
import matplotlib.pyplot as plt

image = []
label = []

def get_files(file_dir):
    for file in os.listdir(file_dir + '\\' + '1'):
        image.append(file_dir + '\\' + '1' + '\\' + file)
        label.append(0)
    for file in os.listdir(file_dir + '\\' + '2'):
        image.append(file_dir + '\\' + '2' + '\\' + file)
        label.append(1)
    for file in os.listdir(file_dir + '\\' + '3'):
        image.append(file_dir + '\\' + '3' + '\\' + file)
        label.append(2)
    for file in os.listdir(file_dir + '\\' + '4'):
        image.append(file_dir + '\\' + '4' + '\\' + file)
        label.append(3)
    for file in os.listdir(file_dir + '\\' + '5'):
        image.append(file_dir + '\\' + '5' + '\\' + file)
        label.append(4)
    for file in os.listdir(file_dir + '\\' + '6'):
        image.append(file_dir + '\\' + '6' + '\\' + file)
        label.append(5)
    for file in os.listdir(file_dir + '\\' + '7'):
        image.append(file_dir + '\\' + '7' + '\\' + file)
        label.append(6)
    for file in os.listdir(file_dir + '\\' + '8'):
        image.append(file_dir + '\\' + '8' + '\\' + file)
        label.append(7)
    for file in os.listdir(file_dir + '\\' + '9'):
        image.append(file_dir + '\\' + '9' + '\\' + file)
        label.append(8)
    for file in os.listdir(file_dir + '\\' + '10'):
        image.append(file_dir + '\\' + '10' + '\\' + file)
        label.append(9)


    # 打乱数据集
    temp = np.array([image, label])
    temp = temp.transpose()
    np.random.shuffle(temp)#行和行打乱

    # 分出image和label
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list


train_dir = r'E:\近期作业\AI_HW\raw_data'
image_list, label_list = get_files(train_dir)
#initialization
Train_image = np.random.rand(len(image_list) - 200, 64, 64, 3).astype('float32')
Train_label = np.random.rand(len(image_list) - 200, 1).astype('int')
Test_image = np.random.rand(200, 64, 64, 3).astype('float32')
Test_label = np.random.rand(200, 1).astype('int')

for i in range(len(image_list) - 200):
    Train_image[i] = np.array(plt.imread(image_list[i]))
    Train_label[i] = np.array(label_list[i])

for i in range(len(image_list) - 200, len(image_list)):
    Test_image[i + 200 - len(image_list)] = np.array(plt.imread(image_list[i]))
    Test_label[i + 200 - len(image_list)] = np.array(label_list[i])

f = h5py.File('data.h5', 'w')
f.create_dataset('X_train', data=Train_image)
f.create_dataset('y_train', data=Train_label)
f.create_dataset('X_test', data=Test_image)
f.create_dataset('y_test', data=Test_label)
f.close()
