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