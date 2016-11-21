#encoding: utf-8
import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_classes', 10, 'the number of classes.')
flags.DEFINE_float('learning_rate_decay_factor', 0.9, 'learning rate decay factor.')
flags.DEFINE_float('learning_rate', 1e-2, 'initial learning rate.')
flags.DEFINE_string('tower_name', 'tower', 'gpu tower name.')