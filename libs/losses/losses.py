# -*- coding: utf-8 -*-
"""
@author: jemmy li
@contact: zengarden2009@gmail.com
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def _smooth_l1_loss_base(bbox_pred, bbox_targets, sigma=1.0):
    '''

    :param bbox_pred: [-1, 4] in RPN. [-1, cls_num+1, 4] in Fast-rcnn
    :param bbox_targets: shape is same as bbox_pred
    :param sigma:
    :return:
    '''
    sigma_2 = sigma**2

    box_diff = bbox_pred - bbox_targets

    abs_box_diff = tf.abs(box_diff)

    smoothL1_sign = tf.stop_gradient(
        tf.to_float(tf.less(abs_box_diff, 1. / sigma_2)))
    loss_box = tf.pow(box_diff, 2) * (sigma_2 / 2.0) * smoothL1_sign \
               + (abs_box_diff - (0.5 / sigma_2)) * (1.0 - smoothL1_sign)
    return loss_box



#定义Giou loss和IoU loss函数
def giou_loss(y_true, y_pred, mode='giou'):
    if mode not in ['giou', 'iou']:
        raise ValueError("Value of mode should be 'iou' or 'giou'")
    y_pred = tf.convert_to_tensor(y_pred)
    if not y_pred.dtype.is_floating:
        y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, y_pred.dtype)
    giou = _giou_loss(y_pred, y_true, mode)

    return 1 - giou

def _giou_loss(b1, b2, mode='giou'):
    zero = tf.convert_to_tensor(0., b1.dtype)

    #unstack 坐标的值，注意这个地方是已经把dx，dy，dw，dh变成了left top bottum down的形式
    b1_xmin, b1_ymin, b1_xmax, b1_ymax = tf.unstack(b1, 4, axis=-1)
    b2_xmin, b2_ymin, b2_xmax, b2_ymax = tf.unstack(b2, 4, axis=-1)


    b1_width = tf.maximum(zero, b1_xmax - b1_xmin)
    b1_height = tf.maximum(zero, b1_ymax - b1_ymin)
    b2_width = tf.maximum(zero, b2_xmax - b2_xmin)
    b2_height = tf.maximum(zero, b2_ymax - b2_ymin)
    b1_area = b1_width * b1_height
    b2_area = b2_width * b2_height

    intersect_ymin = tf.maximum(b1_ymin, b2_ymin)
    intersect_xmin = tf.maximum(b1_xmin, b2_xmin)
    intersect_ymax = tf.minimum(b1_ymax, b2_ymax)
    intersect_xmax = tf.minimum(b1_xmax, b2_xmax)
    intersect_width = tf.maximum(zero, intersect_xmax - intersect_xmin)
    intersect_height = tf.maximum(zero, intersect_ymax - intersect_ymin)
    intersect_area = intersect_width * intersect_height

    union_area = b1_area + b2_area - intersect_area

    #防止益处
    iou = tf.math.divide_no_nan(intersect_area, union_area)
    if mode == 'iou':
        return iou

    enclose_ymin = tf.minimum(b1_ymin, b2_ymin)
    enclose_xmin = tf.minimum(b1_xmin, b2_xmin)
    enclose_ymax = tf.maximum(b1_ymax, b2_ymax)
    enclose_xmax = tf.maximum(b1_xmax, b2_xmax)
    enclose_width = tf.maximum(zero, enclose_xmax - enclose_xmin)
    enclose_height = tf.maximum(zero, enclose_ymax - enclose_ymin)
    enclose_area = enclose_width * enclose_height
    giou = iou - tf.math.divide_no_nan(
        (enclose_area - union_area), enclose_area)
    return giou

def decode_boxes():
    pass



def smooth_l1_loss_rpn(bbox_pred, bbox_targets, label, sigma=1.0):
    '''

    :param bbox_pred: [-1, 4]
    :param bbox_targets: [-1, 4]
    :param label: [-1]
    :param sigma:
    :return:
    '''
    #value = _smooth_l1_loss_base(bbox_pred, bbox_targets, sigma=sigma)
    value = giou_loss(bbox_pred, bbox_targets)
    print('*************************************')
    print("value: ", value)
    #print('value_iou: ', value_iou)
    #value = tf.reduce_sum(value, axis=1)  # to sum in axis 1
    #print('value_reduce: ', value)
    rpn_positive = tf.where(tf.greater(label, 0))

    # rpn_select = tf.stop_gradient(rpn_select) # to avoid
    selected_value = tf.gather(value, rpn_positive)
    non_ignored_mask = tf.stop_gradient(
        1.0 - tf.to_float(tf.equal(label, -1)))  # positve is 1.0 others is 0.0

    bbox_loss = tf.reduce_sum(selected_value) / tf.maximum(1.0, tf.reduce_sum(non_ignored_mask))

    return bbox_loss

def smooth_l1_loss_rcnn(bbox_pred, bbox_targets, label, num_classes, sigma=1.0):
    '''

    :param bbox_pred: [-1, (cfgs.CLS_NUM +1) * 4]
    :param bbox_targets:[-1, (cfgs.CLS_NUM +1) * 4]
    :param label:[-1]
    :param num_classes:
    :param sigma:
    :return:
    '''

    outside_mask = tf.stop_gradient(tf.to_float(tf.greater(label, 0)))

    bbox_pred = tf.reshape(bbox_pred, [-1, num_classes, 4])
    bbox_targets = tf.reshape(bbox_targets, [-1, num_classes, 4])

    value = _smooth_l1_loss_base(bbox_pred, bbox_targets, sigma=sigma)
    #value = giou_loss(bbox_pred, bbox_targets)
    print('*************************************')
    print("value: ", value)
    value = tf.reduce_sum(value, 2)
    #print('value_reduce: ', value)
    value = tf.reshape(value, [-1, num_classes])

    inside_mask = tf.one_hot(tf.reshape(label, [-1, 1]),
                             depth=num_classes, axis=1)

    inside_mask = tf.stop_gradient(
        tf.to_float(tf.reshape(inside_mask, [-1, num_classes])))

    normalizer = tf.to_float(tf.shape(bbox_pred)[0])
    bbox_loss = tf.reduce_sum(
        tf.reduce_sum(value * inside_mask, 1)*outside_mask) / normalizer

    return bbox_loss

def sum_ohem_loss(cls_score, label, bbox_pred, bbox_targets,
                  nr_ohem_sampling, nr_classes, sigma=1.0):

    raise NotImplementedError('not implement Now. YJR will implemetn in the future')
