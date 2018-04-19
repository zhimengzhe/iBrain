# -*- coding: utf-8 -*-
# -----------------------------
# 获取数据
# Author: 樊亚磊
# Date: 2017.10.09
# -----------------------------
import tensorflow as tf
import numpy as np
import os

class Info:

    def getImageFiles(self, file_dir, model = {}):
        image_list = []
        label_list = []
        for file in os.listdir(file_dir):
            name = file.split('.')
            image_list.append(file_dir + file)
            label_list.append(model[name[0]])
        temp = np.array([image_list, label_list])
        temp = temp.transpose()
        np.random.shuffle(temp)
        image_list = list(temp[:, 0])
        label_list = list(temp[:, 1])
        label_list = [int(i) for i in label_list]
        return image_list, label_list

    def getImageBatch(self, image, label, image_W = 208, image_H = 208, batch_size = 16, capacity = 2000):
        '''
        Args:
            image: list type
            label: list type
            image_W: image width
            image_H: image height
            batch_size: batch size
            capacity: the maximum elements in queue
        Returns:
            image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
            label_batch: 1D tensor [batch_size], dtype=tf.int32
        '''

        image = tf.cast(image, tf.string)
        label = tf.cast(label, tf.int32)

        # make an input queue
        input_queue = tf.train.slice_input_producer([image, label])

        label = input_queue[1]
        image_contents = tf.read_file(input_queue[0])
        image = tf.image.decode_jpeg(image_contents, channels=3)

        ######################################
        # data argumentation should go to here
        ######################################

        image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)

        # if you want to test the generated batches of images, you might want to comment the following line.
        image = tf.image.per_image_standardization(image)

        image_batch, label_batch = tf.train.batch([image, label],
                                                  batch_size=batch_size,
                                                  num_threads=64,
                                                  capacity=capacity)

        # you can also use shuffle_batch
        #    image_batch, label_batch = tf.train.shuffle_batch([image,label],
        #                                                      batch_size=BATCH_SIZE,
        #                                                      num_threads=64,
        #                                                      capacity=CAPACITY,
        #                                                      min_after_dequeue=CAPACITY-1)

        label_batch = tf.reshape(label_batch, [batch_size])
        image_batch = tf.cast(image_batch, tf.float32)

        return image_batch, label_batch

    def getImages(self, train_dir, mode = {}, image_W = 208, image_H = 208, batch_size = 16, capacity = 2000):
        train, train_label = self.getImageFiles(train_dir, mode)
        return self.getImageBatch(train, train_label, image_W, image_H, batch_size, capacity)

    def getOneImage(self, img_dir):
        from PIL import Image
        import matplotlib.pyplot as plt
        image = Image.open(img_dir)
        plt.imshow(image)
        image = image.resize([208, 208])
        image = np.array(image)
        return image

    def relight(self, img, light=1, bias=0):
        w = img.shape[1]
        h = img.shape[0]
        # image = []
        for i in range(0, w):
            for j in range(0, h):
                for c in range(3):
                    tmp = int(img[j, i, c] * light + bias)
                    if tmp > 255:
                        tmp = 255
                    elif tmp < 0:
                        tmp = 0
                    img[j, i, c] = tmp
        return img

    def getMyFace(self):
        import cv2
        import dlib
        import sys
        import random

        output_dir = './data/face/my_faces'
        size = 64

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        detector = dlib.get_frontal_face_detector()
        camera = cv2.VideoCapture(0)

        index = 1
        while True:
            if (index <= 200):
                print('Being processed picture %s' % index)
                success, img = camera.read()
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                dets = detector(gray_img, 1)
                for i, d in enumerate(
                        dets):  # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中
                    x1 = d.top() if d.top() > 0 else 0
                    y1 = d.bottom() if d.bottom() > 0 else 0
                    x2 = d.left() if d.left() > 0 else 0
                    y2 = d.right() if d.right() > 0 else 0
                    face = img[x1:y1, x2:y2]
                    face = self.relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
                    face = cv2.resize(face, (size, size))
                    cv2.imshow('image', face)
                    cv2.imwrite(output_dir + '/' + str(index) + '.jpg', face)
                    index += 1
                key = cv2.waitKey(30) & 0xff
                if key == 27:
                    break
            else:
                print('Finished!')
                break

    def setOtherFace(self):
        import sys
        import cv2
        import dlib

        input_dir = './data/face/input_img'
        output_dir = './data/face/other_faces'
        size = 64
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        detector = dlib.get_frontal_face_detector()
        index = 1
        for (path, dirnames, filenames) in os.walk(input_dir):
            for filename in filenames:
                if filename.endswith('.jpg'):
                    print('Being processed picture %s' % index)
                    img_path = path + '/' + filename
                    # 从文件读取图片
                    img = cv2.imread(img_path)
                    # 转为灰度图片
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # 使用detector进行人脸检测 dets为返回的结果
                    dets = detector(gray_img, 1)

                    # 使用enumerate 函数遍历序列中的元素以及它们的下标
                    # 下标i即为人脸序号
                    # left：人脸左边距离图片左边界的距离 ；right：人脸右边距离图片左边界的距离
                    # top：人脸上边距离图片上边界的距离 ；bottom：人脸下边距离图片上边界的距离
                    for i, d in enumerate(dets):
                        x1 = d.top() if d.top() > 0 else 0
                        y1 = d.bottom() if d.bottom() > 0 else 0
                        x2 = d.left() if d.left() > 0 else 0
                        y2 = d.right() if d.right() > 0 else 0
                        # img[y:y+h,x:x+w]
                        face = img[x1:y1, x2:y2]
                        # 调整图片的尺寸
                        face = cv2.resize(face, (size, size))
                        cv2.imshow('image', face)
                        # 保存图片
                        cv2.imwrite(output_dir + '/' + str(index) + '.jpg', face)
                        index += 1

                    key = cv2.waitKey(30) & 0xff
                    if key == 27:
                        sys.exit(0)