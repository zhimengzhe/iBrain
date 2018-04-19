# -*- coding: utf-8 -*-
# -----------------------------
# 人脸识别
# Author: 樊亚磊
# Date: 2018.4.19
# -----------------------------
import tensorflow as tf
import cv2, dlib
import numpy as np
import os
import random
import sys
from sklearn.model_selection import train_test_split

MAX_STEP = 10000 # with current parameters, it is suggested to use MAX_STEP>10k
LEARN_RATE = 0.0001 # with current parameters, it is suggested to use learning rate<=0.0001
SIZE = 64

class Face:

    my_faces_path = './data/face/my_faces'
    other_faces_path = './data/face/other_faces'

    def __init__(self):
        self.imgs = []
        self.labs = []
        self.readData(self.my_faces_path)
        self.readData(self.other_faces_path)
        self.imgs = np.array(self.imgs)
        self.labs = np.array([[0, 1] if lab == self.my_faces_path else [1, 0] for lab in self.labs])
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.imgs, self.labs, test_size=0.05,
                                                            random_state=random.randint(0, 100))
        self.train_x = self.train_x.reshape(self.train_x.shape[0], SIZE, SIZE, 3)
        self.test_x = self.test_x.reshape(self.test_x.shape[0], SIZE, SIZE, 3)
        self.train_x = self.train_x.astype('float32') / 255.0
        self.test_x = self.test_x.astype('float32') / 255.0

        print('train size:%s, test size:%s' % (len(self.train_x), len(self.test_x)))
        self.batch_size = 100
        self.num_batch = len(self.train_x) // self.batch_size

        self.x = tf.placeholder(tf.float32, [None, SIZE, SIZE, 3])
        self.y_ = tf.placeholder(tf.float32, [None, 2])

        self.keep_prob_5 = tf.placeholder(tf.float32)
        self.keep_prob_75 = tf.placeholder(tf.float32)

    def getPaddingSize(self, img):
        h, w, _ = img.shape
        top, bottom, left, right = (0, 0, 0, 0)
        longest = max(h, w)

        if w < longest:
            tmp = longest - w
            left = tmp // 2
            right = tmp - left
        elif h < longest:
            tmp = longest - h
            top = tmp // 2
            bottom = tmp - top
        else:
            pass
        return top, bottom, left, right

    def readData(self, path, h=SIZE, w=SIZE):
        for filename in os.listdir(path):
            if filename.endswith('.jpg'):
                filename = path + '/' + filename
                img = cv2.imread(filename)
                top, bottom, left, right = self.getPaddingSize(img)
                img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                img = cv2.resize(img, (h, w))  # 图片缩放，统一标准

                self.imgs.append(img)
                self.labs.append(path)

    def weightVariable(self, shape):
        init = tf.random_normal(shape, stddev=0.01)
        return tf.Variable(init)

    def biasVariable(self, shape):
        init = tf.random_normal(shape)
        return tf.Variable(init)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def maxPool(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def dropout(self, x, keep):
        return tf.nn.dropout(x, keep)

    def cnnLayer(self):
        # 第一层
        W1 = self.weightVariable([3, 3, 3, 32])  # 卷积核大小(3,3)， 输入通道(3)， 输出通道(32)
        b1 = self.biasVariable([32])
        # 卷积
        conv1 = tf.nn.relu(self.conv2d(self.x, W1) + b1)
        # 池化
        pool1 = self.maxPool(conv1)
        # 减少过拟合，随机让某些权重不更新
        drop1 = self.dropout(pool1, self.keep_prob_5)

        # 第二层
        W2 = self.weightVariable([3, 3, 32, 64])
        b2 = self.biasVariable([64])
        conv2 = tf.nn.relu(self.conv2d(drop1, W2) + b2)
        pool2 = self.maxPool(conv2)
        drop2 = self.dropout(pool2, self.keep_prob_5)

        # 第三层
        W3 = self.weightVariable([3, 3, 64, 64])
        b3 = self.biasVariable([64])
        conv3 = tf.nn.relu(self.conv2d(drop2, W3) + b3)
        pool3 = self.maxPool(conv3)
        drop3 = self.dropout(pool3, self.keep_prob_5)

        # 全连接层
        Wf = self.weightVariable([8 * 8 * 64, 512])
        bf = self.biasVariable([512])
        drop3_flat = tf.reshape(drop3, [-1, 8 * 8 * 64])
        dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
        dropf = self.dropout(dense, self.keep_prob_75)

        # 输出层
        Wout = self.weightVariable([512, 2])
        bout = self.weightVariable([2])
        # out = tf.matmul(dropf, Wout) + bout
        out = tf.add(tf.matmul(dropf, Wout), bout)
        return out

    def faceTrain(self):
        out = self.cnnLayer()

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=self.y_))

        train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(self.y_, 1)), tf.float32))
        tf.summary.scalar('loss', cross_entropy)
        tf.summary.scalar('accuracy', accuracy)
        merged_summary_op = tf.summary.merge_all()
        saver = tf.train.Saver()

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            summary_writer = tf.summary.FileWriter('./log/face/train', graph=tf.get_default_graph())

            for n in range(10):
                for i in range(self.num_batch):
                    batch_x = self.train_x[i * self.batch_size: (i + 1) * self.batch_size]
                    batch_y = self.train_y[i * self.batch_size: (i + 1) * self.batch_size]
                    _, loss, summary = sess.run([train_step, cross_entropy, merged_summary_op],
                                                feed_dict={self.x: batch_x, self.y_: batch_y, self.keep_prob_5: 0.5,
                                                           self.keep_prob_75: 0.75})
                    summary_writer.add_summary(summary, n * self.num_batch + i)
                    print(n * self.num_batch + i, loss)

                    if (n * self.num_batch + i) % 100 == 0:
                        # 获取测试数据的准确率
                        acc = accuracy.eval({self.x: self.test_x, self.y_: self.test_y, self.keep_prob_5: 1.0, self.keep_prob_75: 1.0})
                        print(n * self.num_batch + i, acc)

                        # 直接保存，没有那么多测试数据
                        saver.save(sess, './log/face/train/train_faces.model', global_step=n * self.num_batch + i)
                        sys.exit(0)
                        # 正常情况应该准确率大于0.98时保存并退出
                        # if acc > 0.98 and n > 2:
                        # saver.save(sess, './train_faces.model', global_step=n*num_batch+i)
                        # sys.exit(0)
            print('accuracy less 0.98, exited!')

    def isMyFace(self):
        output = self.cnnLayer()
        predict = tf.argmax(output, 1)

        saver = tf.train.Saver()
        sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state('./log/face/train')
        if ckpt and ckpt.model_checkpoint_path:
            # saver.restore(sess, tf.train.latest_checkpoint('.'))
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found')
            exit()

        detector = dlib.get_frontal_face_detector()
        cam = cv2.VideoCapture(0)

        while True:
            _, img = cam.read()
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dets = detector(gray_image, 1)
            if not len(dets):
                # print('Can`t get face.')
                cv2.imshow('img', img)
                key = cv2.waitKey(30) & 0xff
                if key == 27:
                    sys.exit(0)

            for i, d in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0
                face = img[x1:y1, x2:y2]
                face = cv2.resize(face, (SIZE, SIZE))
                res = sess.run(predict, feed_dict={self.x: [face / 255.0], self.keep_prob_5: 1.0, self.keep_prob_75: 1.0})
                if res[0] == 1:
                    print('Is this my face? %s' % 'True')
                else:
                    print('Is this my face? %s' % 'False')

                cv2.rectangle(img, (x2, x1), (y2, y1), (255, 0, 0), 3)
                cv2.imshow('image', img)
                key = cv2.waitKey(30) & 0xff
                if key == 27:
                    sys.exit(0)