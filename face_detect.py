from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import json
import copy
import argparse
import facenet
import align.detect_face
import matplotlib.pyplot as plt

# 加载模型参数
def LoadModelParameters(gpu_memory_fraction):
    # 创建网络
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    ModelNet = [pnet, rnet, onet]
    return ModelNet

# 检测人脸位置
def DetectFaceLocation(ImagePath, ModelNet):
    minsize = 20 # 最小人脸尺寸
    threshold = [ 0.6, 0.7, 0.7 ]  # 阈值
    factor = 0.709

    ImgArray = misc.imread(os.path.expanduser(ImagePath), mode='RGB')
    bounding_boxes, _ = align.detect_face.detect_face(ImgArray, minsize, ModelNet[0], ModelNet[1], ModelNet[2], threshold, factor)
    det_result = bounding_boxes.astype(int)
    if len(det_result) < 1:
        return False, None
    return True, det_result

# 生成人脸向量模型
def EmbeddingFace(ImgPath):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # 加载模型
            model = './20181021/'
            facenet.load_model(model)
            # 获取输入输出tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")    

            img = misc.imread(ImgPath, mode='RGB')
            image = facenet.prewhiten(img)
            feed_dict = { images_placeholder: image, phase_train_placeholder:False }
            img_emb = sess.run(embeddings, feed_dict=feed_dict)
    return img_emb

# 标记人脸
def MarkFaces(image, bounding_boxes, margin):
    img_size = np.asarray(image.shape)[0:2]
    det = bounding_boxes

    det[:,0] = np.maximum(det[:,0]-margin/2, 0)
    det[:,1] = np.maximum(det[:,1]-margin/2, 0)
    det[:,2] = np.minimum(det[:,2]+margin/2, img_size[1]-1)
    det[:,3] = np.minimum(det[:,3]+margin/2, img_size[0]-1)

    det = np.rint(det).astype(np.int)
    print(det)
    for i in range(len(det)):
        cv2.rectangle(image, (det[i,0],det[i,1]), (det[i,2],det[i,3]),(0, 255, 0), 5)
    return image

# 向量模型初始化，模型预加载，提高响应速度
# with tf.Graph().as_default():
#     with tf.Session() as sess:
#         # 加载模型
#         model = './20181021/'
#         facenet.load_model(model)
#         # 获取输入输出tensors
#         images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
#         embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
#         phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")    
# EmbeddingModel = [sess, images_placeholder, embeddings, phase_train_placeholder]

# STime = time.time()

# ModelNet = LoadModelParameters(1.0)
# LTime = time.time()
# print("Load time: ", LTime - STime)

# FacesLoc = DetectFaceLocation(img, ModelNet)
# ETime = time.time()
# print("Detect time: ", ETime - LTime)

# FaceImg = MarkFaces(img, FacesLoc, 10)
# plt.imshow(FaceImg)
# plt.show()
