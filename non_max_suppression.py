# -*- coding: utf-8 -*-
import numpy as np

def nms(boxes, threshold):
    '''
    对边界框进行非极大值抑制
    args:
        boxes：边界框，数据为list类型，形状为[n,5]  5位表示(x1,y1,x2,y2,score)
        threshold：IOU阈值  大于该阈值，进行抑制
    '''
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    # 计算边界框区域大小，并按照score进行倒叙排序
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)[::-1]

    # keep为最后保留的边框
    keep = []

    while len(idxs) > 0:
        # idxs[0]是当前分数最大的窗口，肯定保留
        i = idxs[0]
        keep.append(i)

        # 计算窗口i与其他所有窗口的交叠部分的面积
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # 交/并得到iou值
        ovr = inter / (areas[i] + areas[idxs[1:]] - inter)

        # inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
        inds = np.where(ovr <= threshold)[0]

        # order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
        idxs = idxs[inds + 1]

    return boxes[keep]