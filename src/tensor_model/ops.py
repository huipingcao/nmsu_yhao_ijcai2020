# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""The building block ops for Spectral Normalization GAN."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def conv2d(input_, output_dim, k_h=3, k_w=3, d_h=2, d_w=2, name='conv2d'):
    """Creates convolutional layers which use xavier initializer.
    Args:
        input_: 4D input tensor (batch size, height, width, channel).
        output_dim: Number of features in the output layer.
        k_h: The height of the convolutional kernel.
        k_w: The width of the convolutional kernel.
        d_h: The height stride of the convolutional kernel.
        d_w: The width stride of the convolutional kernel.
        name: The name of the variable scope.
    Returns:
        conv: The normalized tensor.
    """
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim], initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.zeros_initializer())
        conv = tf.nn.bias_add(conv, biases)
        return conv



def linear(x, output_size, scope=None, bias_start=0.0):
    """Creates a linear layer.
    Args:
        x: 2D input tensor (batch size, features)
        output_size: Number of features in the output layer
        scope: Optional, variable scope to put the layer's parameters into
        bias_start: The bias parameters are initialized to this value

    Returns:
        The normalized tensor
    """
    shape = x.get_shape().as_list()
    with tf.variable_scope(scope or 'Linear'):
        matrix = tf.get_variable('Matrix', [shape[1], output_size], tf.float32, tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable('bias', [output_size], initializer=tf.constant_initializer(bias_start))
        out = tf.matmul(x, matrix) + bias
        return out


def lrelu(x, leak=0.2, name='lrelu'):
    """The leaky RELU operation."""
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def _l2normalize(v, eps=1e-12):
    """l2 normize the input vector."""
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def spectral_normed_weight(weights, num_iters=1, update_collection=None, name="", with_sigma=False):
    """Performs Spectral Normalization on a weight tensor.
    Specifically it divides the weight tensor by its largest singular value. This is intended to stabilize GAN training, by making the discriminator satisfy a local 1-Lipschitz constraint. Based on [Spectral Normalization for Generative Adversarial Networks][sn-gan]

    Args:
        weights: The weight tensor which requires spectral normalization
        num_iters: Number of SN iterations.
        update_collection: The update collection for assigning persisted variable u.
                       If None, the function will update u during the forward
                       pass. Else if the update_collection equals 'NO_OPS', the
                       function will not update the u during the forward. This
                       is useful for the discriminator, since it does not update
                       u in the second pass.Else, it will put the assignment in a collection
                       defined by the user. Then the user need to run the
                       assignment explicitly.
        with_sigma: For debugging purpose. If True, the fuction returns
                the estimated singular value for the weight tensor.
    Returns:
        w_bar: The normalized weight tensor
        sigma: The estimated singular value for the weight tensor.
    """
    w_shape = weights.shape.as_list()
    w_mat = tf.reshape(weights, [-1, w_shape[-1]])  # [-1, output_channel]
    u = tf.get_variable(name + 'u', [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
    u_ = u
    for _ in range(num_iters):
        v_ = _l2normalize(tf.matmul(u_, w_mat, transpose_b=True))
        u_ = _l2normalize(tf.matmul(v_, w_mat))
    
    sigma = tf.squeeze(tf.matmul(tf.matmul(v_, w_mat), u_, transpose_b=True)) 
    w_mat /= sigma
    if update_collection is None:
        with tf.control_dependencies([u.assign(u_)]):
            w_bar = tf.reshape(w_mat, w_shape)
    else:
        w_bar = tf.reshape(w_mat, w_shape)
        if update_collection != 'NO_OPS':
            tf.add_to_collection(update_collection, u.assign(u_))
    if with_sigma:
        return w_bar, sigma
    else:
        return w_bar


def snconv2d(input_, output_dim, k_h=3, k_w=3, d_h=2, d_w=2, sn_iters=1, update_collection=None, name='snconv2d'):
    """Creates a spectral normalized (SN) convolutional layer.

    Args:
        input_: 4D input tensor (batch size, height, width, channel).
        output_dim: Number of features in the output layer.
        k_h: The height of the convolutional kernel.
        k_w: The width of the convolutional kernel.
        d_h: The height stride of the convolutional kernel.
        d_w: The width stride of the convolutional kernel.
        sn_iters: The number of SN iterations.
        update_collection: The update collection used in spectral_normed_weight.
        name: The name of the variable scope.
    Returns:
        conv: The normalized tensor.

    """
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim], initializer=tf.contrib.layers.xavier_initializer())
        w_bar = spectral_normed_weight(w, num_iters=sn_iters, update_collection=update_collection)

        conv = tf.nn.conv2d(input_, w_bar, strides=[1, d_h, d_w, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.zeros_initializer())
        conv = tf.nn.bias_add(conv, biases)
        return conv


def snlinear(x, output_size, bias_start=0.0, sn_iters=1, update_collection=None, name='snlinear'):
    """Creates a spectral normalized linear layer.

    Args:
        x: 2D input tensor (batch size, features).
        output_size: Number of features in output of layer.
        bias_start: The bias parameters are initialized to this value
        sn_iters: Number of SN iterations.
        update_collection: The update collection used in spectral_normed_weight
        name: Optional, variable scope to put the layer's parameters into
    Returns:
        The normalized tensor
    """
    shape = x.get_shape().as_list()
    with tf.variable_scope(name):
        matrix = tf.get_variable('Matrix', [shape[1], output_size], tf.float32, tf.contrib.layers.xavier_initializer())
        matrix_bar = spectral_normed_weight(matrix, num_iters=sn_iters, update_collection=update_collection)
        bias = tf.get_variable('bias', [output_size], initializer=tf.constant_initializer(bias_start))
        out = tf.matmul(x, matrix_bar) + bias
        return out

