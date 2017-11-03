# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print 'read data finished'

is_training=True

def tf_variable(shape, name=None):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)


def dense_connect(x, shape):
    w = tf_variable(shape)
    b = tf.Variable(tf.zeros([shape[1]]))
    return tf.matmul(x, w) + b


def batch_norm(inputs, is_training, is_conv_out=True, decay=0.999):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
    if is_training:
        if is_conv_out:
            batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
        else:
            batch_mean, batch_var = tf.nn.moments(inputs,[0])

        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, 0.001)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, 0.001)


def conv2d_with_batch_norm(x, filter_shape, stride):

    filter_ = tf_variable(filter_shape)
    conv = tf.nn.conv2d(x, filter=filter_, strides=[1, stride, stride, 1], padding="SAME")
    normed=batch_norm(conv, is_training)

    return  tf.nn.relu(normed)


def conv2d(x, filter_shape, stride):

    out_channels = filter_shape[3]

    conv = tf.nn.conv2d(x, filter=tf_variable(filter_shape), strides=[1, stride, stride, 1], padding="SAME")
    bias = tf.Variable(tf.zeros([out_channels]), name="bias")

    return tf.nn.relu(tf.nn.bias_add(conv,bias))


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def residual_block(x, out_channels, down_sample, projection=False):
    in_channels = x.get_shape().as_list()[3]
    if down_sample:
        x = max_pool(x)

    output = conv2d_with_batch_norm(x, [3, 3, in_channels, out_channels], 1)
    output = conv2d_with_batch_norm(output, [3, 3, out_channels, out_channels], 1)

    if in_channels != out_channels:
        if projection:
            # projection shortcut
            input_ = conv2d(x, [1, 1, in_channels, out_channels], 2)
        else:
            # zero-padding
            input_ = tf.pad(x, [[0,0], [0,0], [0,0], [0, out_channels - in_channels]])
    else:
        input_ = x

    return output + input_


def residual_group(name,x,num_block,out_channels):

    assert num_block>=1,'num_block must greater than 1'

    with tf.variable_scope('%s_head'%name):
        output = residual_block(x, out_channels, True)

    for i in xrange (num_block-1):
        with tf.variable_scope('%s_%d' % (name,i+1)):
            output = residual_block(output,out_channels, False)

    return output


def residual_net(inpt):

    with tf.variable_scope('conv1'):
        output = conv2d(inpt, [3, 3, 1, 16], 1)

    output=residual_group('conv2', x=output,num_block=2,out_channels=16)

    output=residual_group('conv3', x=output,num_block=2,out_channels=32)

    #output=residual_group('conv4', x=output,num_block=2,out_channels=64)

    with tf.variable_scope('fc'):
        output=max_pool(output)

        shape=output.get_shape().as_list()
        i_shape=shape[1]*shape[2]*shape[3]

        output=tf.reshape(output,[-1,i_shape])

        return dense_connect(output, [i_shape, 10])


def train_network(batch_size = 120,training_iters=800,learning_rate=0.001):

    x = tf.placeholder("float", [None, 28, 28, 1])#[batch_size,width,height,channels]
    y = tf.placeholder("float", [None, 10])#[batch_size,num_classes]

    pred = residual_net(x)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracytr = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    accuracyte = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    tf.summary.scalar('cost', cost)
    tf.summary.scalar('train accuracy', accuracytr)
    tf.summary.scalar('test accuracy', accuracyte)

    merged = tf.summary.merge_all()
    init = tf.initialize_all_variables()

    print 'start training...'

    with tf.Session() as sess:
        sess.run(init)
        swriter = tf.summary.FileWriter("log/cancha", sess.graph)
        step = 1
        while step< training_iters:
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            #print np.shape(batch_xs),np.shape(batch_ys)
            batch_xs=np.reshape(batch_xs,[np.shape(batch_xs)[0],28,28,1])

            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            if step % 10 == 0:
                summary,acc = sess.run([merged,accuracytr], feed_dict={x: batch_xs, y: batch_ys})
                swriter.add_summary(summary,step)

                summary,loss = sess.run([merged,cost], feed_dict={x: batch_xs, y: batch_ys})
                swriter.add_summary(summary,step)

                batch_test=mnist.test.images[:256]
                summary,ta=sess.run([merged,accuracyte], feed_dict={x: np.reshape(batch_test,[np.shape(batch_test)[0],28,28,1]), y: mnist.test.labels[:256]})
                swriter.add_summary(summary,step)
                print "%s,loss:%s, train accuracy:%s, test accuray:%s"%(step,"{:.6f}".format(loss),"{:.6f}".format(acc),"{:.6f}".format(ta))

            step += 1
    print "train finished"

if __name__ == '__main__':
    train_network()