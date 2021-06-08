# -*- coding=utf-8 -*-
import glob
import platform
import time
import cv2
from skimage.feature import hog
import numpy as np
import os
import joblib
from sklearn.svm import LinearSVC
import shutil
import sys

from pyramid import pyramid, silding_window
from non_max_suppression import nms

# label
label_map = {-1: 'cat', 1: 'dog'}
# train files path (/DataSet/archive/training_set/training_set)
train_image_path = 'Your test files path'
# test files path (/DataSet/archive/test_set/test_set)
test_image_path = 'Your test files path'

image_height = 100
image_width = 128

train_feat_path = 'train/'
test_feat_path = 'test/'
model_path = 'model/'

cat = 'cat'
dog = 'dog'

train_example = 4000
test_example = 200

# hog获取特征
def obtain_feat(image_list, label_list, savePath):
    i = 0
    for image in image_list:
        try:
            # resize img 128 * 100
            image = cv2.resize(image, (image_width, image_height))
        except:
            print('发送了异常，图片大小size不满足要求', i+1)
            continue
        image = rgb2gray(image)
        fd = hog(image, orientations=12,block_norm='L1', pixels_per_cell=[8, 8], cells_per_block=[4, 4], visualize=False,
                 transform_sqrt=True)
        fd = np.concatenate((fd, [label_list[i]]))
        fd_name = ('cat' if label_list[i] == -1 else 'dog') + str(i) + '.feat'
        fd_path = os.path.join(savePath, fd_name)
        joblib.dump(fd, fd_path)
        print(i)
        i += 1
    print("Test features are extracted and saved.")


# 灰度转换
def rgb2gray(im):
    gray = im[:, :, 0] * 0.2989 + im[:, :, 1] * 0.5870 + im[:, :, 2] * 0.1140
    return gray


# cv2加载图片
def cv2_read_images(method, filePath, test_samples):
    image_list, image_label = [], []
    start = 0 if method == 'train' else 4000
    for index in range(start, start + test_samples):
        cat_path = filePath % (cat+'s', cat, index+1)
        dog_path = filePath % (dog+'s', dog, index+1)
        image_list.append(cv2.imread(cat_path))
        image_list.append(cv2.imread(dog_path))
        image_label.append(-1)
        image_label.append(1)
    return image_list, image_label


# 提取特征
def extra_feat():
    trainFilePath = train_image_path + '/%s/%s.%i.jpg'
    testFilePath = test_image_path + '/%s/%s.%i.jpg'

    train_image_list, train_image_label = cv2_read_images('train', trainFilePath, train_example)
    test_image_list, test_image_label = cv2_read_images('test', testFilePath, test_example)

    obtain_feat(train_image_list, train_image_label, train_feat_path)
    obtain_feat(test_image_list, test_image_label, test_feat_path)


# 创建存放特征与模型的文件夹
def mkdir():
    if not os.path.exists(train_feat_path):
        os.mkdir(train_feat_path)
    if not os.path.exists(test_feat_path):
        os.mkdir(test_feat_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)


# svm训练和测试
def train_and_test():
    t0 = time.time()
    features = []
    labels = []
    correct_number = 0
    total = 0
    for feat_path in glob.glob(os.path.join(train_feat_path, '*.feat')):
        data = joblib.load(feat_path)
        features.append(data[:-1])
        labels.append(data[-1])

    print("Training a Linear LinearSVM Classifier.")
    clf = LinearSVC()
    clf.fit(features, labels)

    # 保存模型

    joblib.dump(clf, model_path + 'model')
    print("Model was saved.")

    # test
    result_list = []
    for feat_path in glob.glob(os.path.join(test_feat_path, '*.feat')):
        total += 1
        if platform.system() == 'Windows':
            symbol = '\\'
        else:
            symbol = '/'
        image_name = feat_path.split(symbol)[1].split('.feat')[0]
        data_test = joblib.load(feat_path)
        data_test_feat = data_test[:-1].reshape((1, -1)).astype(np.float64)
        result = clf.predict(data_test_feat)
        print("Test %s is : %s" % (image_name, label_map[int(result[0])]))
        result_list.append(image_name + ' ' + label_map[int(result[0])] + '\n')
        if int(result[0]) == int(data_test[-1]):
            correct_number += 1
    print("Test was finished.")
    write_to_txt(result_list)
    rate = float(correct_number) / total
    t1 = time.time()
    print('准确率是： %f' % rate)
    print('耗时是 : %f' % (t1 - t0))


def visualization(testPath):
    # 可视化
    img = cv2.imread(testPath)
    img = cv2.resize(img, (image_width, image_height))
    '''
        利用滑动窗口进行目标检测
    '''
    # 滑动窗口大小
    w, h = 96, 76

    rectangles = []
    counter = 1
    scale_factor = 1.5
    font = cv2.FONT_HERSHEY_PLAIN

    # 图像金字塔
    for resized in pyramid(img.copy(), scale_factor, (img.shape[1] // 2, img.shape[1] // 2)):
        print(resized.shape)
        # 图像缩小倍数
        scale = float(img.shape[1]) / float(resized.shape[1])
        # 遍历每一个滑动区域
        for (x, y, roi) in silding_window(resized, 10, (w, h)):
            if roi.shape[1] != w or roi.shape[0] != h:
                continue
            # 这句话根据你的尺寸改改
            fd = hog(img, orientations=12, block_norm='L1', pixels_per_cell=[8, 8], cells_per_block=[4, 4],
                     visualize=False,
                     transform_sqrt=True)
            fd = np.concatenate((fd, [-1]))
            data_test_feat = fd[:-1].reshape((1, -1)).astype(np.float64)
            clf = joblib.load(model_path + 'model')
            label = clf.predict(data_test_feat)
            score = clf.decision_function(data_test_feat)
            # print(label)
            # 识别为狗
            if label == 1:
                # cv2.putText(img, 'Dog Detected', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                # 得分越大，置信度越高
                if score > 0:
                    # print(label,score)
                    # 获取相应边界框的原始大小
                    rx, ry, rx2, ry2 = x * scale, y * scale, (x + w) * scale, (y + h) * scale
                    rectangles.append([rx, ry, rx2, ry2, score])
            counter += 1

    windows = np.array(rectangles)
    boxes = nms(windows, 0.15)

    for x, y, x2, y2, score in boxes:
        cv2.rectangle(img, (int(x), int(y)), (int(x2), int(y2)), (0, 0, 255), 1)
        cv2.putText(img, '%f' % score, (int(x), int(y)), font, 1, (0, 255, 0))

    cv2.imshow('dog_img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def write_to_txt(list):
    with open('result.txt', 'w') as f:
        f.writelines(list)
    print('每张图片的识别结果存放在result.txt里面')


if __name__ == '__main__':

    mkdir()  # 不存在文件夹就创建
    need_extra_feat = input('是否需要重新获取特征？y/n\n')

    if need_extra_feat == 'y':
        shutil.rmtree(train_feat_path)  # rm dir
        shutil.rmtree(test_feat_path)
        mkdir()
        extra_feat()  # 获取特征并保存在文件夹

    train_and_test()  # 训练并预测

    car = './dog.jpg'
    visualization(car)
