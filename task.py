#encoding: utf-8

from util import *
from features import *
import numpy as np
import dataset
import random
from dataset import TrainData
from dataset import TestData
import time
import model
import tensorflow as tf
from tensorflow.python.platform import gfile
from sklearn.metrics import precision_recall_fscore_support
import pickle
# settings
import settings
FLAGS = settings.FLAGS

DATA_HOME = "/mnt/images/radio_data/zenkyoku"
DEBUG_DIR = "output/debug"
AUDIO_EXT = "wav"
AUDIO_NUM = 3

MAX_STEPS = 5000
training_epochs = 1000

CNN = True
BANDS = 60
FRAMES = 41
NUM_INPUT_CHANNELS = 2

train_data_pickle = 'train_cnn_b%d_f%d_c%d.pkl' % (BANDS, FRAMES, NUM_INPUT_CHANNELS)
test_data_pickle = 'test_cnn_b%d_f%d_c%d.pkl' % (BANDS, FRAMES, NUM_INPUT_CHANNELS)

def train():
    if not os.path.isfile(train_data_pickle):
        # trainig data
        train_features, train_labels = features_cnn(["802", "AlphaStation", "fmcocolo", "FMKOBE", "InterFM", "TFM"])
        traindata = TrainData(train_features, train_labels)
        with open(train_data_pickle, mode='wb') as f:
            pickle.dump(traindata, f)
    else:
        print("loading: %s" % (train_data_pickle))
        with open(train_data_pickle, mode='rb') as f:
            traindata = pickle.load(f)
            train_features = traindata.train_inputs
            train_labels = traindata.train_targets

    # if not os.path.isfile(test_data_pickle):
    #     test_features, test_labels = features_cnn(['fold3'])
    #     testdata = TestData(test_features, test_labels)
    #     with open(test_data_pickle, mode='wb') as f:
    #         pickle.dump(testdata, f)
    # else:
    #     print("loading: %s" % (test_data_pickle))
    #     with open(test_data_pickle, mode='rb') as f:
    #         testdata = pickle.load(f)
    #         test_features = testdata.test_inputs
    #         test_labels = testdata.test_targets

    # TODO change to use train and test
    train_labels = one_hot_encode(train_labels)
    #test_labels = one_hot_encode(test_labels)

    # random train and test sets.
    train_test_split = np.random.rand(len(train_features)) < 0.70
    # train:test = 7:3
    train_x = train_features[train_test_split]
    train_y = train_labels[train_test_split]
    test_x = train_features[~train_test_split]
    test_y = train_labels[~train_test_split]

    print("train_x shape: ", train_x.shape)
    x_batches = np.array_split(train_x, 100)
    print("x_batches shape: ", np.array(x_batches).shape)
    print("x_batches[0] shape: ", np.array(x_batches)[0].shape)

    n_dim = train_features.shape[1]
    print("input dim: %s" % (n_dim))

    # create placeholder
    keep_prob = tf.placeholder(tf.float32)
    if CNN:
        X = tf.placeholder(tf.float32, [None, BANDS, FRAMES, NUM_INPUT_CHANNELS])
        Y = tf.placeholder(tf.float32, [None, FLAGS.num_classes])
    else:
        X = tf.placeholder(tf.float32, [None, n_dim])
        Y = tf.placeholder(tf.float32, [None, FLAGS.num_classes])
    # build graph
    if CNN:
        logits = model.cnn_s(X, keep_prob=keep_prob)
    else:
        logits = model.inference(X, n_dim)

    weights = tf.all_variables()
    saver = tf.train.Saver(weights)

    # create loss
    loss = model.loss(logits, Y)
    tf.scalar_summary('loss', loss)

    accracy = model.accuracy(logits, Y)
    tf.scalar_summary('test accuracy', accracy)

    # train operation
    train_op = model.train_op(loss)

    # variable initializer
    init = tf.initialize_all_variables()

    # get Session
    sess = tf.Session()

    # sumary merge and writer
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir)

    # initialize
    sess.run(init)

    print("train_x shape: ", train_x.shape)
    x_batches = np.array_split(train_x, 1000)
    print("x_batches shape: ", np.array(x_batches[0]).shape)
    y_batches = np.array_split(train_y, 1000)

    print("train_x shape: ", test_x.shape)
    test_x_batches = np.array_split(test_x, 1000)
    print("x_batches shape: ", np.array(test_x_batches[0]).shape)
    test_y_batches = np.array_split(test_y, 1000)
    for step in xrange(MAX_STEPS):
        # TODO shuffle batches
        # create batch
        for i, (x_batch, y_batch) in enumerate(zip(x_batches, y_batches)):
            summary, logits_val, _, loss_val = sess.run([merged, logits, train_op, loss], feed_dict={X: np.array(x_batch), Y: np.array(y_batch), keep_prob: 0.5})

            if i % 100 == 0:
                t_pred = sess.run(tf.argmax(logits, 1), feed_dict={X: np.array(x_batch), keep_prob: 0.5})
                t_true = sess.run(tf.argmax(np.array(y_batch), 1))
                print("train samples pred: %s" % t_pred[:30])
                print("train samples target: %s" % t_true[:30])
                print('Train accuracy: ', sess.run(accracy, feed_dict={X: np.array(x_batch), Y: np.array(y_batch), keep_prob: 0.5}))

        train_writer.add_summary(summary, step)
        print("step:%d, loss: %s" % (step, loss_val))

        # choice random number of test batches
        test_rand = random.randint(0, len(test_x_batches))

        y_pred = sess.run(tf.argmax(logits, 1), feed_dict={X: np.array(test_x_batches[test_rand]), keep_prob: 1.0})
        y_true = sess.run(tf.argmax(np.array(test_y_batches[test_rand]), 1))
        print("test samples pred: %s" % y_pred[:10])
        print("test samples target: %s" % y_true[:10])
        accracy_val = sess.run([accracy], feed_dict={X: np.array(test_x_batches[test_rand]), Y: np.array(test_y_batches[test_rand]), keep_prob: 1.0})
        print('Test accuracy: ', accracy_val)
        #train_writer.add_summary(accracy_val, step)
        p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average='micro')
        print("F-score: %s" % f)

        if step % 1000 == 0:
            if not gfile.Exists(FLAGS.ckpt_dir):
                gfile.MakeDirs(FLAGS.ckpt_dir)
            saver.save(sess, os.path.join(FLAGS.ckpt_dir, train_data_pickle), global_step=step)


def features(sub_dirs):
    try:
        features, labels = parse_audio_files(DATA_HOME, sub_dirs)
    except Exception as e:
        print("[Error] parse error. %s" % e)
    return features, labels


def features_cnn(sub_dirs):
    try:
        print("features_cnn")
        features, labels = parse_audio_files_cnn(DATA_HOME, sub_dirs, bands=BANDS, frames=FRAMES, verbose=False)
    except Exception as e:
        print("[Error] parse error. %s" % e)
    return features, labels


def disp_data(names, files):
    plot_waves(names, files)
    plot_specgram(names, files)
    plot_log_power_specgram(names, files)


if __name__ == '__main__':
    print("DATA_HOME: %s" % (DATA_HOME))

    # waves, names = dataset.get_files("802")
    # for wave in waves:
    #     print("="*10)
    #     print("file: %s" % wave)
    #     print_wave_info(wave)
    #     break
    #
    # raw_waves = raw_sounds = load_sound_files(waves)
    # disp_data(names, raw_waves)

    train()



