#encoding: utf-8

from util import *
from features import *
import numpy as np
import dataset
from dataset import TrainData
from dataset import TestData
import time
import model
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
import pickle
# settings
import settings
FLAGS = settings.FLAGS

DATA_HOME = "output/data/UrbanSound8K/audio"
DEBUG_DIR = "output/debug"
AUDIO_EXT = "wav"
AUDIO_NUM = 3

MAX_STEPS = 5000
training_epochs = 100

train_data_pickle = 'train.pkl'
test_data_pickle = 'test.pkl'

def train():
    if not os.path.isfile(train_data_pickle):
        # trainig data
        train_features, train_labels = features(['fold0', 'fold1', 'fold2'])
        traindata = TrainData(train_features, train_labels)
        with open(train_data_pickle, mode='wb') as f:
            pickle.dump(traindata, f)
    else:
        print("loading: %s" % (train_data_pickle))
        with open(train_data_pickle, mode='rb') as f:
            traindata = pickle.load(f)
            train_features = traindata.train_inputs
            train_labels = traindata.train_targets

    if not os.path.isfile(test_data_pickle):
        test_features, test_labels = features(['fold3'])
        testdata = TestData(test_features, test_labels)
        with open(test_data_pickle, mode='wb') as f:
            pickle.dump(testdata, f)
    else:
        print("loading: %s" % (test_data_pickle))
        with open(test_data_pickle, mode='rb') as f:
            testdata = pickle.load(f)
            test_features = testdata.test_inputs
            test_labels = testdata.test_targets


    train_labels = one_hot_encode(train_labels)
    test_labels = one_hot_encode(test_labels)

    train_test_split = np.random.rand(len(train_features)) < 0.70
    train_x = train_features[train_test_split]
    train_y = train_labels[train_test_split]
    test_x = train_features[~train_test_split]
    test_y = train_labels[~train_test_split]


    n_dim = train_features.shape[1]
    print("input dim: %s" % (n_dim))

    # create placeholder
    X = tf.placeholder(tf.float32, [None, n_dim])
    Y = tf.placeholder(tf.float32, [None, FLAGS.num_classes])
    # build graph
    logits = model.inference_org(X, n_dim)

    # create loss
    loss = model.loss(logits, Y)
    accracy = model.accuracy(logits, Y)

    # train operation
    train_op = model.train_op(loss)

    # variable initializer
    init = tf.initialize_all_variables()

    # get Session
    sess = tf.Session()

    # initialize
    sess.run(init)

    for step in xrange(MAX_STEPS):

        t_pred = sess.run(tf.argmax(logits, 1), feed_dict={X: train_features})
        t_true = sess.run(tf.argmax(train_labels, 1))
        print("train samples pred: %s" % t_pred[:30])
        print("train samples target: %s" % t_true[:30])
        print('Train accuracy: ', sess.run(accracy, feed_dict={X: train_x, Y: train_y}))

        start_time = time.time()
        previous_time = start_time
        for epoch in xrange(training_epochs):
            logits_val, _, loss_val = sess.run([logits, train_op, loss], feed_dict={X: train_x, Y: train_y})
        end_time = time.time()
        dutation = end_time - previous_time

        print("step:%d, loss: %s" % (step, loss_val))
        y_pred = sess.run(tf.argmax(logits, 1), feed_dict={X: test_x})
        y_true = sess.run(tf.argmax(test_y, 1))
        print("test samples pred: %s" % y_pred[:10])
        print("test samples target: %s" % y_true[:10])
        print('Test accuracy: ', sess.run(accracy, feed_dict={X: test_x, Y: test_y}))
        p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average='micro')
        print("F-score: %s" % f)


def features(sub_dirs):
    try:
        features, labels = parse_audio_files(DATA_HOME, sub_dirs)
    except Exception as e:
        print("[Error] parse error. %s" % e)
    return features, labels


def disp_data(names, files):
    plot_waves(names, files)
    plot_specgram(names, files)
    plot_log_power_specgram(names, files)


if __name__ == '__main__':
    print("DATA_HOME: %s" % (DATA_HOME))

    waves, names = dataset.get_files("fold1")
    for wave in waves:
        print("="*10)
        print("file: %s" % wave)
        print_wave_info(wave)

    raw_waves = raw_sounds = load_sound_files(waves)
    disp_data(names, raw_waves)

    train()



