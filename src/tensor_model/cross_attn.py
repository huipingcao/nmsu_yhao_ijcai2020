import tensorflow as tf
from .ops import spectral_normed_weight
from .ops import snlinear
from tensorflow.contrib import rnn
#import numpy as np
#import time

#from .model_setting import return_cnn_setting_from_file

#from utils.classification_results import class_based_accuracy
#from sklearn.metrics import accuracy_score


def v_attn_layer(x_4d, name, update_collection=None, init=tf.contrib.layers.xavier_initializer()):
    with tf.variable_scope(name):
        in_chan = x_4d.get_shape().as_list()[3]
        out_chan = in_chan//8
        if out_chan == 0:
            out_chan = 1
        query_conv = sn_conv1x1(x_4d, out_chan, update_collection, init, 'v_attn_query')  # B * T * A * C1
        key_conv = sn_conv1x1(x_4d, out_chan, update_collection, init, 'v_attn_key')  # B * T * A * C1
        #key_conv = tf.layers.max_pooling2d(inputs=key_conv, pool_size=[2, 1], strides=[2, 1])  # downsampling on the key, reduce overall dimension
        value_conv = sn_conv1x1(x_4d, in_chan // 2, update_collection, init, 'v_attn_value')  # B * T * A * C2
        #value_conv = tf.layers.max_pooling2d(inputs=value_conv, pool_size=[2, 1], strides=[2, 1])
        #print(x_4d.get_shape())
        #print(query_conv.get_shape())
        #print(key_conv.get_shape())
        #print(value_conv.get_shape())
        attn = tf.matmul(query_conv, key_conv, transpose_b=True)  # trainspose key_conv to be B * T * C1 * A, output is B * T * A * A
        attn = tf.nn.softmax(attn)
        #print(attn.get_shape())
        #print(value_conv.get_shape())
        attn_value = tf.matmul(attn, value_conv)  # output is B * T * A * C2
        #print(attn_value.get_shape())
        sigma = tf.get_variable('sigma_ratio', [], initializer=tf.constant_initializer(0.0))
        attn_value = sn_conv1x1(attn_value, in_chan, update_collection, init, 'v_attn_attn')  # B * T * A * C
        #print(attn_value.get_shape())
        #print((x_4d + sigma * attn_value).get_shape())
        return x_4d + sigma * attn_value, attn

def t_attn_layer(x_4d, name, only_before=True, update_collection=None, init=tf.contrib.layers.xavier_initializer()):
    with tf.variable_scope(name):
        in_chan = x_4d.get_shape().as_list()[3]
        out_chan = in_chan//8
        if out_chan == 0:
            out_chan = 1
        query_conv = sn_conv1x1(x_4d, out_chan, update_collection, init, 't_attn_query')  # B * T * A * C1
        query_conv = tf.transpose(query_conv, perm=[0, 2, 1, 3])  # B * A * T * C1
        key_conv = sn_conv1x1(x_4d, out_chan, update_collection, init, 't_attn_key')  # B * T * A * C1
        key_conv = tf.transpose(key_conv, perm=[0, 2, 1, 3])  # B * A * T * C1
        #key_conv = tf.layers.max_pooling2d(inputs=key_conv, pool_size=[2, 1], strides=[2, 1])  # downsampling on the key, reduce overall dimension
        value_conv = sn_conv1x1(x_4d, in_chan // 2, update_collection, init, 't_attn_value')  # B * T * A * C2
        value_conv = tf.transpose(value_conv, perm=[0, 2, 1, 3])  # B * A * T * C2
        #value_conv = tf.layers.max_pooling2d(inputs=value_conv, pool_size=[2, 1], strides=[2, 1])
        #print(x_4d.get_shape())
        #print(query_conv.get_shape())
        #print(key_conv.get_shape())
        #print(value_conv.get_shape())
        attn = tf.matmul(query_conv, key_conv, transpose_b=True)  # trainspose key_conv to be B * A * C1 * T, output is B * A * T * T
        if only_before is True:
            attn = tf.linalg.LinearOperatorLowerTriangular(attn).to_dense()
        attn = tf.nn.softmax(attn)
        #print(attn.get_shape())
        #print(value_conv.get_shape())
        attn_value = tf.matmul(attn, value_conv)  # output is B * A * T * C2
        #print(attn_value.get_shape())
        sigma = tf.get_variable('sigma_ratio', [], initializer=tf.constant_initializer(0.0))
        attn_value = sn_conv1x1(attn_value, in_chan, update_collection, init, 't_attn_attn')  # B * A * T * C
        attn_value = tf.transpose(attn_value, perm=[0, 2, 1, 3])  # B * T * A * C
        #print(attn_value.get_shape())
        #print((x_4d + sigma * attn_value).get_shape())
        #sdf
        return x_4d + sigma * attn_value, attn

# class prediction based attention layer, applied before the last softmax layer
def c_attn_layer(logits_out, fun_name="c_attn", update_collection=None, init=tf.contrib.layers.xavier_initializer()):
    with tf.variable_scope(fun_name):
        batch_num, num_classes = logits_out.get_shape().as_list()
        feature_num = 100
        query_linear = snlinear(logits_out, feature_num, update_collection=update_collection, name=fun_name+'_query')
        key_linear = snlinear(logits_out, feature_num, update_collection=update_collection, name=fun_name+"_key")
        value_linear = snlinear(logits_out, feature_num, update_collection=update_collection, name=fun_name+"_value")

        print(query_linear.get_shape())
        print(key_linear.get_shape())
        print(value_linear.get_shape())

        attn = tf.matmul(query_linear, key_linear, transpose_a=True)
        attn = tf.nn.softmax(attn)
        print(attn.get_shape())

        attn_value = tf.matmul(value_linear, attn, transpose_b=True)  # output is B * A * T * C2
        print(attn_value.get_shape())
        sigma = tf.get_variable('sigma_ratio', [], initializer=tf.constant_initializer(0.0))
        return x_2d + sigma * attn_value, attn


def global_attn_layer(x_4d, name, update_collection=None, init=tf.contrib.layers.xavier_initializer()):
    with tf.variable_scope(name):
        b_num, f_num, v_num, in_chan = x_4d.get_shape().as_list()
        b_num = -1 if type(b_num)==type(None) else b_num
        all_num = f_num * v_num
        out_chan = in_chan//8
        if out_chan == 0:
            out_chan = 1
        query_conv = sn_conv1x1(x_4d, out_chan, update_collection, init, name +'global_attn_query')  # B * T * A * C1
        query_conv = tf.reshape(query_conv, [b_num, all_num, out_chan])

        key_conv = sn_conv1x1(x_4d, out_chan, update_collection, init, name + 'global_attn_key')  # B * T * A * C1
        key_conv = tf.layers.max_pooling2d(inputs=key_conv, pool_size=[2, 1], strides=[2, 1])
        key_b_num, key_f_num, key_v_num, key_out_chan = key_conv.get_shape().as_list()
        downsampled_num = key_f_num * key_v_num
        #print(key_conv.get_shape())
        key_conv = tf.reshape(key_conv, [b_num, downsampled_num, out_chan])
        #print(key_conv.get_shape())
        value_conv = sn_conv1x1(x_4d, in_chan // 2, update_collection, init, name + 'global_attn_value')  # B * T * A * C2
        value_conv = tf.layers.max_pooling2d(inputs=value_conv, pool_size=[2, 1], strides=[2, 1])
        value_conv = tf.reshape(value_conv, [b_num, downsampled_num, in_chan // 2])

        #print(query_conv.get_shape())
        #print(key_conv.get_shape())
        #print(value_conv.get_shape())

        attn = tf.matmul(query_conv, key_conv, transpose_b=True)  # trainspose key_conv to be B * T * C1 * A, output is B * T * A * A
        attn = tf.nn.softmax(attn)
        #print(attn.get_shape())
        #sdfsd
        attn_value = tf.matmul(attn, value_conv)  # output is B * T * A * C2
        #print(attn_value.get_shape())
        attn_value = tf.reshape(attn_value, [b_num, f_num, v_num, in_chan//2])
        #print(attn_value.get_shape())
        sigma = tf.get_variable('sigma_ratio', [], initializer=tf.constant_initializer(0.0))
        attn_value = sn_conv1x1(attn_value, in_chan, update_collection, init, 'v_attn_attn')  # B * T * A * C
        #print(attn_value.get_shape())
        #print((x_4d + sigma * attn_value).get_shape())
        
        return x_4d + sigma * attn_value, attn



# Convert to only one channel
def input_attn_layer(x_4d, name, update_collection=None, init=tf.contrib.layers.xavier_initializer()):
    """
    " This is the attention mechanism using in https://arxiv.org/pdf/1704.02971.pdf
    """
    with tf.variable_scope(name):
        b_num, t_num, v_num, c_chan = x_4d.get_shape().as_list()
        b_num = -1 if type(b_num)==type(None) else b_num
        f_num = c_chan * v_num
        hidden_num = f_num //8
        #print(x_4d.get_shape())
        variable_count = 0
        x_3d = sn_conv1x1(x_4d, 1, update_collection, init, name+"_" + str(variable_count))
        variable_count = variable_count + 1
        x_3d = tf.reshape(x_3d, [b_num, t_num, v_num])
        #print(x_3d.get_shape())
        # Define a lstm cell with tensorflow
        lstm_cell = rnn.BasicLSTMCell(hidden_num, forget_bias=1.0)
        # Get lstm cell output
        init_state = lstm_cell.zero_state(tf.shape(x_4d)[0], dtype=tf.float32)
        state = init_state
        attn_matrix = []
        reuse_bool = False
        for timestep in range(t_num):
            if timestep > 0:
                reuse_bool = True
            with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_bool):
                x_t = x_3d[:, timestep, :]
                # 这里的state保存了每一层 LSTM 的状态
                (cell_output, state) = lstm_cell(x_t, state)
                #print(cell_output.get_shape())
                #print(state.c.get_shape())
                #print(state.h.get_shape())
                #print(x_t.get_shape())
                c_state = state.c
                h_state = state.h
                all_state = tf.concat([x_t, c_state, h_state], 1)
                attn_t = snlinear(all_state, 1, update_collection=update_collection)
                variable_count = variable_count + 1
                attn_t = tf.nn.softmax(attn_t)
                attn_t = tf.tile(attn_t, [1, v_num])
                attn_matrix.append(attn_t)
        attn_tensor = tf.convert_to_tensor(attn_matrix)
        attn_tensor = tf.transpose(attn_tensor, [1, 0, 2])
        #print(111)
        #print(x_3d.get_shape())
        #print(attn_tensor.get_shape())
        x_3d_attn = tf.math.multiply(x_3d, attn_tensor)
        x_4d_attn = tf.reshape(x_3d_attn, [b_num, t_num, v_num, 1])
        #print(x_4d_attn.get_shape())
        #sdfds
        #x_4d_out = sn_conv1x1(x_4d_attn, 128, update_collection, init)
        x_4d_out = sn_conv1x1(x_4d_attn, 128, update_collection, init, name=name+"_" + str(variable_count))
        variable_count = variable_count + 1
        #print(x_4d_out.get_shape())
        #print(222)
        #sdfsd
        return x_4d_out

# with all channel
def input_attn_layer_with_all_channel(x_4d, name, update_collection=None, init=tf.contrib.layers.xavier_initializer()):
    """
    " This is the attention mechanism using in https://arxiv.org/pdf/1704.02971.pdf
    """
    with tf.variable_scope(name):
        b_num, t_num, v_num, c_chan = x_4d.get_shape().as_list()
        b_num = -1 if type(b_num)==type(None) else b_num
        f_num = c_chan * v_num
        hidden_num = f_num //8
        #print(x_4d.get_shape())
        variable_count = 0
        
        # Define a lstm cell with tensorflow
        lstm_cell = rnn.BasicLSTMCell(hidden_num, forget_bias=1.0)
        # Get lstm cell output
        init_state = lstm_cell.zero_state(tf.shape(x_4d)[0], dtype=tf.float32)
        state = init_state
        attn_matrix = []
        reuse_bool = False
        for chan_i in range(c_chan):
            x_chan = x_4d[:, :, :, chan_i]
            attn_chan_matrix = []
            for timestep in range(t_num):
                if timestep > 0:
                    reuse_bool = True
                with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_bool):
                    x_t = x_chan[:, timestep, :]
                    # 这里的state保存了每一层 LSTM 的状态
                    (cell_output, state) = lstm_cell(x_t, state)
                    #print(cell_output.get_shape())
                    #print(state.c.get_shape())
                    #print(state.h.get_shape())
                    #print(x_t.get_shape())
                    c_state = state.c
                    h_state = state.h
                    all_state = tf.concat([x_t, c_state, h_state], 1)
                    attn_t = snlinear(all_state, 1, update_collection=update_collection)
                    variable_count = variable_count + 1
                    attn_t = tf.nn.softmax(attn_t)
                    attn_t = tf.tile(attn_t, [1, v_num])
                    attn_chan_matrix.append(attn_t)
            attn_matrix.append(attn_chan_matrix)
            #if chan_i > 3:
            #    break
        #print(x_4d.get_shape())
        attn_tensor = tf.convert_to_tensor(attn_matrix)
        attn_tensor = tf.transpose(attn_tensor, [2, 1, 3, 0])
        #print(111)
        #print(x_4d.get_shape())
        #print(attn_tensor.get_shape())
        #sdfsd
        x_4d_out = tf.math.multiply(x_4d, attn_tensor)
        #print(x_4d_attn.get_shape())
        #sdfds
        #x_4d_out = sn_conv1x1(x_4d_attn, 128, update_collection, init)
        #x_4d_out = sn_conv1x1(x_4d_attn, 128, update_collection, init, name=name+"_" + str(variable_count))
        #variable_count = variable_count + 1
        #print(x_4d_out.get_shape())
        #print(222)
        #sdfsd
        return x_4d_out



def rec_attn_layer(x_4d, name, update_collection=None, init=tf.contrib.layers.xavier_initializer()):
    with tf.variable_scope(name):
        batch_size, a_num, f_len, in_chan = x_4d.get_shape().as_list()


def sn_non_local_block_sim(x, update_collection, name, init=tf.contrib.layers.xavier_initializer()):
    with tf.variable_scope(name):
        batch_size, h, w, num_channels = x.get_shape().as_list()
        location_num = h * w
        downsampled_num = location_num // 4
        # theta path
        theta = sn_conv1x1(x, num_channels // 8, update_collection, init, 'sn_conv_theta')
        theta = tf.reshape(theta, [batch_size, location_num, num_channels // 8])
        # phi path
        phi = sn_conv1x1(x, num_channels // 8, update_collection, init, 'sn_conv_phi')
        phi = tf.layers.max_pooling2d(inputs=phi, pool_size=[2, 2], strides=2)
        phi = tf.reshape(phi, [batch_size, downsampled_num, num_channels // 8])
        attn = tf.matmul(theta, phi, transpose_b=True)
        attn = tf.nn.softmax(attn)
        print(tf.reduce_sum(attn, axis=-1))
        # g path
        g = sn_conv1x1(x, num_channels // 2, update_collection, init, 'sn_conv_g')
        g = tf.layers.max_pooling2d(inputs=g, pool_size=[2, 2], strides=2)
        g = tf.reshape(g, [batch_size, downsampled_num, num_channels // 2])
        attn_g = tf.matmul(attn, g)
        attn_g = tf.reshape(attn_g, [batch_size, h, w, num_channels // 2])
        sigma = tf.get_variable('sigma_ratio', [], initializer=tf.constant_initializer(0.0))
        attn_g = sn_conv1x1(attn_g, num_channels, update_collection, init, 'sn_conv_attn')
        return x + sigma * attn_g


def conv1x1(input_, output_dim, init=tf.contrib.layers.xavier_initializer(), name='conv1x1'):
    k_h = 1
    k_w = 1
    d_h = 1
    d_w = 1
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim], initializer=init)
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
    return conv

def sn_conv1x1(input_, output_dim, update_collection, init=tf.contrib.layers.xavier_initializer(), name="sn_conv1x1"):
    with tf.variable_scope(name):
        k_h = 1
        k_w = 1
        d_h = 1
        d_w = 1
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim], initializer=init)
        w_bar = spectral_normed_weight(w, num_iters=1, update_collection=update_collection, name="w_bar")

    conv = tf.nn.conv2d(input_, w_bar, strides=[1, d_h, d_w, 1], padding='SAME')
    return conv



if __name__ == "__main__":
    print("main")
