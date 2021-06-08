# -*- coding: utf-8 -*-
import numpy as np
import cv2

def resize(img, scale_factor):
    '''
    对图像进行缩放

    args：
        img：输入图像
        scale_factor：缩放因子 缩小scale_factor>1
    '''
    ret = cv2.resize(img, (int(img.shape[1] * (1.0 / scale_factor)), int(img.shape[0] * (1.0 / scale_factor))),
                     interpolation=cv2.INTER_AREA)
    return ret


def pyramid(img, scale=1.5, min_size=(200, 200)):
    '''
    图像金字塔 对图像进行缩放，这是一个生成器
    args：
        img：输入图像
        scale：缩放因子
        min_size：图像缩放的最小尺寸 (w,h)
    '''
    yield img
    while True:
        img = resize(img, scale)
        if img.shape[0] < min_size[1] or img.shape[1] < min_size[0]:
            break
        yield img


def silding_window(img, stride, window_size):
    '''
    滑动窗口函数，给定一张图像，返回一个从左到右滑动的窗口，直至覆盖整个图像的宽度，然后回到左边界
    继续下一个步骤，直至覆盖图像的宽度，这样反复进行，直至到图像的右下角
    args：
        img：输入图像
        stride：滑动步长  标量
        widow_size:(w,h) 一定不能大于img大小
    return:
        返回滑动窗口：x,y,滑动区域图像
    '''
    for y in range(0, img.shape[0] - window_size[1], stride):
        for x in range(0, img.shape[1] - window_size[0], stride):
            yield (x, y, img[y:y + window_size[1], x:x + window_size[0]])