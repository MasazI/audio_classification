#encoding: utf-8
import tensorflow as tf
from tensorflow.python.training import moving_averages

# settings
import settings
FLAGS = settings.FLAGS

NUM_CLASSES = FLAGS.num_classes
LEARNING_RATE_DECAY_FACTOR = FLAGS.learning_rate_decay_factor
INITIAL_LEARNING_RATE = FLAGS.learning_rate

# multiple GPU's prefix
TOWER_NAME = FLAGS.tower_name

# Used to keep the update ops done by batch_norm.
UPDATE_OPS_COLLECTION = '_update_ops_'

def _variable_with_weight_decay(name, shape, stddev, wd, trainable=True):
    '''
    重み減衰を利用した変数の初期化
    '''
    var = _variable_on_gpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _variable_on_gpu(name, shape, initializer):
    '''
    GPUメモリに変数をストアする
    '''
    #with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
    return var


def conv2d(scope_name, inputs, shape, bias_shape, stride, padding='VALID', wd=0.0, reuse=False, trainable=True):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        kernel = _variable_with_weight_decay(
            'weights',
            shape=shape,
            stddev=0.01,
            wd=0.0,  # not use weight decay
            trainable=trainable
        )
        conv = tf.nn.conv2d(inputs, kernel, stride, padding=padding)
        #biases = _variable_on_gpu('biases', bias_shape, tf.constant_initializer(0.1), trainable=trainable)
        #bias = tf.nn.bias_add(conv, biases)
        bn = batch_norm(conv, trainable=trainable)
        conv_ = tf.nn.relu(bn, name=scope.name)
        return conv_


def fc_sigmmoid(scope_name, inputs, shape, bias_shape=None, reuse=False, trainable=True, batchn=False):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        weights = _variable_with_weight_decay(
            'weights',
            shape,
            stddev=0.01,
            wd=0.04,
            trainable=trainable
        )
        flat = tf.reshape(inputs, [-1, shape[0]])
        if batchn:
            fc = tf.matmul(flat, weights)
            bn = batch_norm(fc)
            fc = tf.nn.sigmoid(bn, name=scope.name)
        else:
            biases = _variable_on_gpu('biases', bias_shape, tf.constant_initializer(0.1))
            fc = tf.nn.sigmoid(tf.matmul(flat, weights) + biases)
        return fc


def fc_tanh(scope_name, inputs, shape, bias_shape=None, reuse=False, trainable=True, batchn=False):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        weights = _variable_with_weight_decay(
            'weights',
            shape,
            stddev=0.01,
            wd=0.04,
            trainable=trainable
        )
        flat = tf.reshape(inputs, [-1, shape[0]])
        if batchn:
            fc = tf.matmul(flat, weights)
            bn = batch_norm(fc)
            fc = tf.nn.tanh(bn, name=scope.name)
        else:
            biases = _variable_on_gpu('biases', bias_shape, tf.constant_initializer(0.1))
            fc = tf.nn.tanh(tf.matmul(flat, weights) + biases)
        return fc


def fc(scope_name, inputs, shape, bias_shape=None, reuse=False, trainable=True, batchn=False):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        weights = _variable_with_weight_decay(
            'weights',
            shape,
            stddev=0.01,
            wd=0.04,
            trainable=trainable
        )
        flat = tf.reshape(inputs, [-1, shape[0]])
        if batchn:
            fc = tf.matmul(flat, weights)
            bn = batch_norm(fc)
            fc = tf.nn.relu(bn, name=scope.name)
        else:
            biases = _variable_on_gpu('biases', bias_shape, tf.constant_initializer(0.1))
            fc = tf.nn.relu_layer(flat, weights, biases, name=scope.name)

        return fc


def fc_softmax(scope_name, inputs, shape, bias_shape=None, reuse=False, trainable=True, batchn=False):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        flat = tf.reshape(inputs, [-1, shape[0]])
        weights = _variable_with_weight_decay(
            'weights',
            shape,
            stddev=0.01,
            wd=0.04,
            trainable=trainable
        )
        if batchn:
            fc = tf.matmul(flat, weights)
            bn = batch_norm(fc)
            fc = tf.nn.softmax(bn, name=scope.name)
        else:
            biases = _variable_on_gpu('biases', bias_shape, tf.constant_initializer(0.1))
            fc = tf.nn.softmax(tf.matmul(inputs, weights) + biases, name=scope.name)

        return fc


def batch_norm(inputs,
                       decay=0.999,
                       center=True,
                       scale=False,
                       epsilon=0.001,
                       moving_vars='moving_vars',
                       activation=None,
                       is_training=True,
                       trainable=True,
                       restore=True,
                       scope=None,
                       reuse=None):
    """Adds a Batch Normalization layer.
    Args:
      inputs: a tensor of size [batch_size, height, width, channels]
              or [batch_size, channels].
      decay: decay for the moving average.
      center: If True, subtract beta. If False, beta is not created and ignored.
      scale: If True, multiply by gamma. If False, gamma is
        not used. When the next layer is linear (also e.g. ReLU), this can be
        disabled since the scaling can be done by the next layer.
      epsilon: small float added to variance to avoid dividing by zero.
      moving_vars: collection to store the moving_mean and moving_variance.
      activation: activation function.
      is_training: whether or not the model is in training mode.
      trainable: whether or not the variables should be trainable or not.
      restore: whether or not the variables should be marked for restore.
      scope: Optional scope for variable_op_scope.
      reuse: whether or not the layer and its variables should be reused. To be
        able to reuse the layer scope must be given.
    Returns:
      a tensor representing the output of the operation.
    """
    inputs_shape = inputs.get_shape()
    with tf.variable_op_scope([inputs], scope, 'BatchNorm', reuse=reuse):
        axis = list(range(len(inputs_shape) - 1))
        params_shape = inputs_shape[-1:]
        # Allocate parameters for the beta and gamma of the normalization.
        beta, gamma = None, None
        if center:
            beta = tf.get_variable('beta',
                                      params_shape,
                                      initializer=tf.zeros_initializer,
                                      trainable=trainable)
        if scale:
            gamma = tf.get_variable('gamma',
                                       params_shape,
                                       initializer=tf.ones_initializer,
                                       trainable=trainable)
        # 移動平均と移動分散を作成する(明示的にリストアが必要)
        # Create moving_mean and moving_variance add them to
        # GraphKeys.MOVING_AVERAGE_VARIABLES collections. (restoreに使う)
        moving_collections = [moving_vars, tf.GraphKeys.MOVING_AVERAGE_VARIABLES]
        moving_mean = tf.get_variable('moving_mean',
                                         params_shape,
                                         initializer=tf.zeros_initializer,
                                         trainable=False)
        moving_variance = tf.get_variable('moving_variance',
                                             params_shape,
                                             initializer=tf.ones_initializer,
                                             trainable=False)

        if is_training:
            # Calculate the moments based on the individual batch.
            mean, variance = tf.nn.moments(inputs, axis)

            update_moving_mean = moving_averages.assign_moving_average(
                moving_mean, mean, decay)
            tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
            update_moving_variance = moving_averages.assign_moving_average(
                moving_variance, variance, decay)
            tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)
        else:
            # Just use the moving_mean and moving_variance.
            mean = moving_mean
            variance = moving_variance
        # Normalize the activations.
        outputs = tf.nn.batch_normalization(
            inputs, mean, variance, beta, gamma, epsilon)
        outputs.set_shape(inputs.get_shape())
        if activation:
            outputs = activation(outputs)
        return outputs