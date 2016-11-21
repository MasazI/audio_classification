#encoding: utf-8
import tensorflow as tf
import numpy as np
from model_parts import *

fc1_hidden = 280
fc2_hidden = 300

def inference(inputs, n_dim, reuse=False, trainable=True):
    # scope_name, inputs, shape, bias_shape=None, reuse=False, trainable=True
    fc1_output = fc_tanh("fc1", inputs, [n_dim, fc1_hidden], [fc1_hidden], reuse, trainable)
    fc2_output = fc_sigmmoid("fc2", fc1_output, [fc1_hidden, fc2_hidden], [fc2_hidden], reuse, trainable)
    softmax_linear = fc_softmax("softmax", fc2_output, [fc2_hidden, FLAGS.num_classes], [FLAGS.num_classes], reuse, trainable)
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

