#encoding: utf-8
import tensorflow as tf
import numpy as np
from model_parts import *

fc1_hidden = 280
fc2_hidden = 300

# cnn_s
conv1_input_channel_size = 2
conv1_kernel_size = 30
conv1_filter_size = 20
fc1_hidden = 300
fc2_hidden = 300

# cnn_m
conv1_m_input_channel_size = 2
conv1_m_kernel_size_h = 57
conv1_m_kernel_size_w = 6
conv1_m_filter_size = 80
conv1_m_stride = [1, 1, 1, 1]
pool1_m_kernel_size_h = 4
pool1_m_kernel_size_w = 3
pool1_m_stride = [1, 1, 3, 1]
conv2_m_kernel_size_h = 1
conv2_m_kernel_size_w = 3
conv2_m_stride = [1, 1, 3, 1]
fc1_m_hidden = 4096
fc2_m_hidden = 4096

def inference(inputs, n_dim, reuse=False, trainable=True):
    # scope_name, inputs, shape, bias_shape=None, reuse=False, trainable=True
    fc1_output = fc_tanh("fc1", inputs, [n_dim, fc1_hidden], [fc1_hidden], reuse, trainable)

    fc2_output = fc_sigmmoid("fc2", fc1_output, [fc1_hidden, fc2_hidden], [fc2_hidden], reuse, trainable)
    softmax_linear = fc_softmax("softmax", fc2_output, [fc2_hidden, FLAGS.num_classes], [FLAGS.num_classes], reuse, trainable)
    return softmax_linear

def cnn_s(inputs, reuse=False, trainable=True, keep_prob=0.5):
    # conv2d(scope_name, inputs, shape, bias_shape, stride, padding='VALID', wd=0.0, reuse=False, trainable=True)
    conv1 = conv2d('conv1', inputs, [conv1_kernel_size, conv1_kernel_size, conv1_input_channel_size, conv1_filter_size], [conv1_filter_size], [1,1,1,1], padding='SAME', reuse=reuse)
    conv1_dropout = tf.nn.dropout(conv1, keep_prob=0.5)
    conv1_shape = conv1_dropout.get_shape().as_list()
    print conv1_shape
    #fc1 = fc_sigmmoid('fc1', conv1, [conv1_shape[1]*conv1_shape[2]*conv1_shape[3], fc1_hidden], [fc1_hidden], reuse=reuse)
    fc1 = fc_sigmmoid('fc1', conv1_dropout, [60*41*20, fc1_hidden], [fc1_hidden], reuse=reuse)
    softmax_linear = fc_softmax("softmax", fc1, [fc2_hidden, FLAGS.num_classes], [FLAGS.num_classes], reuse, trainable)
    return softmax_linear

def cnn_m(inputs, reuse=False, trainable=True):
    # conv1_m_input_channel_size = 2
    # conv1_m_kernel_size_h = 57
    # conv1_m_kernel_size_w = 6
    # conv1_m_filter_size = 80
    # conv1_m_stride = [1, 1, 1, 1]
    # pool1_m_kernel_size_h = 4
    # pool1_m_kernel_size_w = 3
    # pool1_m_stride = [1, 1, 3, 1]
    # conv2_m_kernel_size_h = 1
    # conv2_m_kernel_size_w = 3
    # conv2_m_stride = [1, 1, 3, 1]
    # fc1_m_hidden = 4096
    # fc2_m_hidden = 4096

    # conv2d(scope_name, inputs, shape, bias_shape, stride, padding='VALID', wd=0.0, reuse=False, trainable=True)
    conv1 = conv2d('conv1', inputs, [conv1_m_kernel_size_h, conv1_m_kernel_size_w, conv1_m_input_channel_size, conv1_m_filter_size], [conv1_m_filter_size], conv1_m_stride, padding='SAME', reuse=reuse)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, pool1_m_kernel_size_h, pool1_m_kernel_size_w, 1], strides=pool1_m_stride, padding='SAME', name='pool1')
    pool1_dropout = tf.nn.dropout(pool1, keep_prob=0.5)
    pool1_dropout_shape = pool1_dropout.get_shape().as_list()
    #fc1 = fc_sigmmoid('fc1', conv1, [conv1_shape[1]*conv1_shape[2]*conv1_shape[3], fc1_hidden], [fc1_hidden], reuse=reuse)
    fc1 = fc_sigmmoid('fc1', conv1, [60*41*20, fc1_hidden], [fc1_hidden], reuse=reuse)
    softmax_linear = fc_softmax("softmax", fc1, [fc2_hidden, FLAGS.num_classes], [FLAGS.num_classes], reuse, trainable)
    return softmax_linear

def loss(logits, targets):
    print("logits shape: %s" % logits.get_shape())
    print("targets shape: %s" % targets.get_shape())
    loss = tf.reduce_mean(-tf.reduce_sum(targets*tf.log(logits), reduction_indices=[1]))
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, targets))
    return loss


def accuracy(logits, targets):
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(targets, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy


def train_op(loss):
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)
    return optimizer

