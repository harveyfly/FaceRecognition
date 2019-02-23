from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import time
from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
import facenet
import align.detect_face
import random

from os.path import join as pjoin
import matplotlib.pyplot as plt

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics  
from sklearn.externals import joblib

def main():
    with tf.Graph().as_default():
        with tf.Session() as sess:     
            # 加载模型
            model='./20181021/'
            facenet.load_model(model)
            # 获取输入输出tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            image = []
            nrof_images = 0
            # 这里要改为自己emb_img文件夹的位置
            emb_dir = './emb_img'
            all_obj = []
            for i in os.listdir(emb_dir):
                all_obj.append(i)
                img = misc.imread(os.path.join(emb_dir,i), mode='RGB')
                prewhitened = facenet.prewhiten(img)
                image.append(prewhitened)
                nrof_images = nrof_images+1

            images = np.stack(image)
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            compare_emb = sess.run(embeddings, feed_dict=feed_dict) 
            compare_num = len(compare_emb)

# 修改版load_and_align_data
# 传入rgb np.ndarray
def load_and_align_data(img, image_size, margin):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    img_size = np.asarray(img.shape)[0:2]

    # bounding_boxes shape:(1,5)  type:np.ndarray
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    # 如果未发现目标 直接返回
    if len(bounding_boxes) < 1:
        return 0,0,0

    # 从数组的形状中删除单维度条目，即把shape中为1的维度去掉
    # det = np.squeeze(bounding_boxes[:,0:4])
    det = bounding_boxes

    print('det shape type')
    print(det.shape)
    print(type(det))

    det[:,0] = np.maximum(det[:,0]-margin/2, 0)
    det[:,1] = np.maximum(det[:,1]-margin/2, 0)
    det[:,2] = np.minimum(det[:,2]+margin/2, img_size[1]-1)
    det[:,3] = np.minimum(det[:,3]+margin/2, img_size[0]-1)

    det = det.astype(int)
    crop = []
    for i in range(len(bounding_boxes)):
        temp_crop = img[det[i,1]:det[i,3],det[i,0]:det[i,2],:]
        aligned = misc.imresize(temp_crop, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        crop.append(prewhitened)

    # np.stack 将crop由一维list变为二维
    crop_image = np.stack(crop)  

    return 1,det,crop_image