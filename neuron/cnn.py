# -*- coding: utf-8 -*-
# -----------------------------
# CNN算法
# Author: 樊亚磊
# Date: 2017.8.31
# -----------------------------
import tensorflow as tf
import os
import numpy as np
MAX_STEP = 10000 # with current parameters, it is suggested to use MAX_STEP>10k
LEARN_RATE = 0.0001 # with current parameters, it is suggested to use learning rate<=0.0001

class Cnn:

    def __init__(self, action):
        self.logs_train_dir = action

    def inference(self, images, batch_size, n_classes):
        '''Build the model
        Args:
            images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
        Returns:
            output tensor with the computed logits, float, [batch_size, n_classes]
        '''
        # conv1, shape = [kernel size, kernel size, channels, kernel numbers]

        with tf.variable_scope('conv1') as scope:
            weights = tf.get_variable('weights',
                                      shape=[3, 3, 3, 16],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
            biases = tf.get_variable('biases',
                                     shape=[16],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)

        # pool1 and norm1
        with tf.variable_scope('pooling1_lrn') as scope:
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                   padding='SAME', name='pooling1')
            norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                              beta=0.75, name='norm1')

        # conv2
        with tf.variable_scope('conv2') as scope:
            weights = tf.get_variable('weights',
                                      shape=[3, 3, 16, 16],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
            biases = tf.get_variable('biases',
                                     shape=[16],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation, name='conv2')

        # pool2 and norm2
        with tf.variable_scope('pooling2_lrn') as scope:
            norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                              beta=0.75, name='norm2')
            pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1],
                                   padding='SAME', name='pooling2')

        # local3
        with tf.variable_scope('local3') as scope:
            reshape = tf.reshape(pool2, shape=[batch_size, -1])
            dim = reshape.get_shape()[1].value
            weights = tf.get_variable('weights',
                                      shape=[dim, 128],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
            biases = tf.get_variable('biases',
                                     shape=[128],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

            # local4
        with tf.variable_scope('local4') as scope:
            weights = tf.get_variable('weights',
                                      shape=[128, 128],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
            biases = tf.get_variable('biases',
                                     shape=[128],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')

        # softmax
        with tf.variable_scope('softmax_linear') as scope:
            weights = tf.get_variable('softmax_linear',
                                      shape=[128, n_classes],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
            biases = tf.get_variable('biases',
                                     shape=[n_classes],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.1))
            softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')

        return softmax_linear

    def losses(self, logits, labels):
        '''Compute loss from logits and labels
        Args:
            logits: logits tensor, float, [batch_size, n_classes]
            labels: label tensor, tf.int32, [batch_size]

        Returns:
            loss tensor of float type
        '''
        with tf.variable_scope('loss') as scope:
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits \
                (logits=logits, labels=labels, name='xentropy_per_example')
            loss = tf.reduce_mean(cross_entropy, name='loss')
            tf.summary.scalar(scope.name + '/loss', loss)
        return loss

    def trainning(self, loss, learning_rate):
        '''Training ops, the Op returned by this function is what must be passed to 
            'sess.run()' call to cause the model to train.

        Args:
            loss: loss tensor, from losses()

        Returns:
            train_op: The op for trainning
        '''
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    def evaluation(self, logits, labels):
        """Evaluate the quality of the logits at predicting the label.
        Args:
          logits: Logits tensor, float - [batch_size, NUM_CLASSES].
          labels: Labels tensor, int32 - [batch_size], with values in the
            range [0, NUM_CLASSES).
        Returns:
          A scalar int32 tensor with the number of examples (out of batch_size)
          that were predicted correctly.
        """
        with tf.variable_scope('accuracy') as scope:
            correct = tf.nn.in_top_k(logits, labels, 1)
            correct = tf.cast(correct, tf.float16)
            accuracy = tf.reduce_mean(correct)
            tf.summary.scalar(scope.name + '/accuracy', accuracy)
        return accuracy

    def trainCnnNetwork(self, train_batch, train_label_batch, n_class = 2, batch_size = 16):
        train_logits = self.inference(train_batch, batch_size, n_class)
        train_loss = self.losses(train_logits, train_label_batch)
        train_op = self.trainning(train_loss, LEARN_RATE)
        train_acc = self.evaluation(train_logits, train_label_batch)
        summary_op = tf.summary.merge_all()
        sess = tf.Session()
        train_writer = tf.summary.FileWriter(self.logs_train_dir, sess.graph)
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                    break
                _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])

                if step % 50 == 0:
                    print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
                    summary_str = sess.run(summary_op)
                    train_writer.add_summary(summary_str, step)

                if step % 2000 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(self.logs_train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()

        coord.join(threads)
        sess.close()

    def evaluateOneImage(self, image_array, n_class = 2, mode = {}):
        mode = {value: key for key, value in mode.items()}
        with tf.Graph().as_default():
            image = tf.cast(image_array, tf.float32)
            image = tf.image.per_image_standardization(image)
            image = tf.reshape(image, [1, 208, 208, 3])
            logit = self.inference(image, 1, n_class)
            logit = tf.nn.softmax(logit)
            x = tf.placeholder(tf.float32, shape=[208, 208, 3])
            saver = tf.train.Saver()
            with tf.Session() as sess:
                print("Reading checkpoints...")
                ckpt = tf.train.get_checkpoint_state(self.logs_train_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('Loading success, global_step is %s' % global_step)
                else:
                    print('No checkpoint file found')

                prediction = sess.run(logit, feed_dict={x: image_array})
                for i in range(len(prediction[0])):
                    print(mode[i] + ' with possibility %.6f' % prediction[:, i])