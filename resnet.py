# Coder: Wang Pei
# Github: https://github.com/xiaokeai18/R3D
# ==============================================================================
'''
This is the resnet structure
'''
import numpy as np
import tensorflow as tf
from hyper_parameters import *


BN_EPSILON = 0.001

def activation_summary(x):
    '''
    :param x: A Tensor
    :return: Add histogram summary and scalar summary of the sparsity of the tensor
    '''
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
    '''
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
    layers.
    :return: The created variable
    '''
    
    ## TODO: to allow different weight decay to fully connected layer and conv layer
    regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)

    new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                    regularizer=regularizer)
    return new_variables


def output_layer(input_layer, num_labels):
    '''
    :param input_layer: 2D tensor
    :param num_labels: int. How many output labels in total? 
    :return: output layer Y = WX + B
    '''
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True,#initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
                            initializer=tf.initializers.variance_scaling(scale=1.0,distribution="uniform"))
    fc_b = create_variables(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer())

    fc_h = tf.matmul(input_layer, fc_w) + fc_b
    return fc_h


def batch_normalization_layer(input_layer, dimension):
    '''
    Helper function to do batch normalziation
    :param input_layer: 5D tensor, [batch, in_depth, in_height, in_width, in_channels]
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
    :return: the 4D tensor after being normalized
    '''
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2, 3],keep_dims=True)
    beta = tf.get_variable('beta', dimension, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', dimension, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

    return bn_layer


def conv_bn_relu_layer(input_layer, filter_shape, stride):
    '''
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_layer: 5D tensor, [batch, in_depth, in_height, in_width, in_channels]
    :param filter_shape: list. [filter_depth, filter_height, filter_width, in_channels, filter_number]
    :param stride: The stride of the sliding window for each dimension of input. Must have strides[0] = strides[4] = 1.
    :return: 5D tensor. Has the same type as input. Y = Relu(batch_normalize(conv(X)))
    '''

    out_channel = filter_shape[-1]
    filter = create_variables(name='conv', shape=filter_shape)

    conv_layer = tf.nn.conv3d(input_layer, filter, strides=[1, stride, stride, stride, 1], padding='SAME')
    bn_layer = batch_normalization_layer(conv_layer, out_channel)

    output = tf.nn.relu(bn_layer)
    return output


def bn_relu_conv_layer(input_layer, filter_shape, stride):
    '''
    A helper function to batch normalize, relu and conv the input layer sequentially
    :param input_layer: 5D tensor, [batch, in_depth, in_height, in_width, in_channels]
    :param filter_shape: list. [filter_depth, filter_height, filter_width, in_channels, out_channels]
    :param stride: stride size for each dimension of `input`
    :return: 5D tensor. Y = conv(Relu(batch_normalize(X)))
    '''

    in_channel = input_layer.get_shape().as_list()[-1]

    bn_layer = batch_normalization_layer(input_layer, in_channel)
    relu_layer = tf.nn.relu(bn_layer)

    filter = create_variables(name='conv', shape=filter_shape)
    conv_layer = tf.nn.conv3d(relu_layer, filter, strides=[1, stride, stride, stride, 1], padding='SAME')
    return conv_layer



def residual_block(input_layer, output_channel, first_block=False):
    '''
    Defines a residual block in ResNet
    :param input_layer: 5D tensor, [batch, in_depth, in_height, in_width, in_channels]
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param first_block: if this is the first residual block of the whole network
    :return: 5D tensor.
    '''
    input_channel = input_layer.get_shape().as_list()[-1]

    # When it's time to "shrink" the image size, we use stride = 2
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('conv1_in_block'):
        if first_block:
            filter = create_variables(name='conv', shape=[3, 3, 3, input_channel, output_channel])
            conv1 = tf.nn.conv3d(input_layer, filter=filter, strides=[1, 1, 1, 1, 1], padding='SAME')
        else:
            conv1 = bn_relu_conv_layer(input_layer, [3, 3, 3, input_channel, output_channel], stride)

    with tf.variable_scope('conv2_in_block'):
        conv2 = bn_relu_conv_layer(conv1, [3, 3, 3, output_channel, output_channel], 1)

    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers
    if increase_dim is True:
        pooled_input = tf.nn.avg_pool3d(input_layer, ksize=[1, 2, 2, 2, 1],
                                      strides=[1, 2, 2, 2, 1], padding='SAME')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                         input_channel // 2]])
    else:
        padded_input = input_layer

    output = conv2 + padded_input
    return output


def inference(input_tensor_batch, n, reuse):
    '''
    The main function that defines the ResNet. total layers = 1 + 2n + 2n + 4n + 2n +1 = 10n + 2
    :param input_tensor_batch: 5D tensor, which is [batch, in_depth, in_height, in_width, in_channels]
    :param n: num_residual_blocks
    :param reuse: To build train graph, reuse=False. To build validation graph and share weights
    with train graph, resue=True
    :return: last layer in the network. Not softmax-ed
    '''

    layers = []
    with tf.variable_scope('conv0', reuse=reuse):
        conv0 = conv_bn_relu_layer(input_tensor_batch, [3, 3, 3, 3, 16], 1)
        #activation_summary(conv0)
        layers.append(conv0)

    for i in range(n):
        with tf.variable_scope('conv1_%d' %i, reuse=reuse):
            if i == 0:
                conv1 = residual_block(layers[-1], 16, first_block=True)
            else:
                conv1 = residual_block(layers[-1], 16)
            #activation_summary(conv1)
            layers.append(conv1)
    print(layers[-1].get_shape())
    for i in range(n):
        with tf.variable_scope('conv2_%d' %i, reuse=reuse):
            conv2 = residual_block(layers[-1], 32)
            #activation_summary(conv2)
            layers.append(conv2)
    print(layers[-1].get_shape())
    for i in range(n):
        with tf.variable_scope('conv3_%d' %i, reuse=reuse):
            conv3 = residual_block(layers[-1], 64)
            layers.append(conv3)
        assert conv3.get_shape().as_list()[1:] == [6, 15, 20, 64]
    print(layers[-1].get_shape())
    for i in range(n):
        with tf.variable_scope('conv4_%d' %i, reuse=reuse):
            conv4 = residual_block(layers[-1], 128)
            layers.append(conv4)
        with tf.variable_scope('conv5_%d' %i, reuse=reuse):    
            conv5 = residual_block(layers[-1], 128)
            layers.append(conv5)
        assert conv5.get_shape().as_list()[1:] == [3, 8, 10, 128]
    print(layers[-1].get_shape())
    for i in range(n):
        with tf.variable_scope('conv6_%d' %i, reuse=reuse):
            conv6 = residual_block(layers[-1], 256)
            layers.append(conv6)
        assert conv6.get_shape().as_list()[1:] == [2, 4, 5, 256]
    print(layers[-1].get_shape())

    with tf.variable_scope('fc', reuse=reuse):
        in_channel = layers[-1].get_shape().as_list()[-1]
        bn_layer = batch_normalization_layer(layers[-1], in_channel)
        relu_layer = tf.nn.relu(bn_layer)
        global_pool = tf.reduce_mean(relu_layer, [1, 2, 3])

        assert global_pool.get_shape().as_list()[-1:] == [256]
        output = output_layer(global_pool, FLAGS.NUM_CLASSES)
        layers.append(output)

    return layers[-1]


def test_graph(train_dir='logs'):
    '''
    Run this function to look at the graph structure on tensorboard. A fast way!
    :param train_dir:
    '''
    input_tensor = tf.constant(np.ones([10, 24, 32, 32, 3]), dtype=tf.float32)
    result = inference(input_tensor, 2, reuse=False)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

#test_graph(train_dir=train_dir)