import tensorflow as tf
import numpy as np
import time

from .model_setting import return_cnn_setting_from_file
from .cross_attn import v_attn_layer
from .cross_attn import t_attn_layer
from .cross_attn import global_attn_layer
from .cross_attn import input_attn_layer
from .cross_attn import input_attn_layer_with_all_channel

#from .cross_attn import c_attn_layer
from .ops import spectral_normed_weight
from .ops import snlinear

from utils.classification_results import class_based_accuracy
#from sklearn.metrics import accuracy_score
#import os
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

class cnn_model_class:
    data_group = None

    learning_rate = 4e-4
    train_x_place = None
    #Here the train_y_place and all other y related places are matrix
    train_y_place = None
    valid_x_place = None
    valid_y_place = None
    dropout_place = None

    predict_y_proba = None
    cross_entropy = None
    coefficient_place = None
    attn_list = []
    logger = None
    data_stru = None
    cnn_setting = None
    saver_file = ''
    cnn_session = None
    config = None

    def __init__(self, data_group, cnn_setting, logger):
        self.data_group = data_group
        self.cnn_setting = cnn_setting
        self.logger = logger
        self.data_stru = data_group.gene_data_stru()
        if cnn_setting.learning_rate > 0:
            self.learning_rate = cnn_setting.learning_rate
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        #self.config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True, gpu_options=gpu_options)
        self.config = tf.ConfigProto(allow_soft_placement=True)
        #self.config.gpu_options.allow_growth = True

    def cnn_graph_setup(self, data_stru, cnn_setting, input_map=1):
        logger = self.logger
        attr_num = data_stru.attr_num
        attr_len = data_stru.attr_len
        num_classes = data_stru.num_classes

        tf.reset_default_graph()
        tf.random.set_random_seed(0)
        self.train_x_place = tf.placeholder(tf.float32, [None, attr_len, attr_num, input_map])
        self.train_y_place = tf.placeholder(tf.float32, [None, num_classes])

        last_conv_out, saver_file, layer, self.is_train_place, self.attn_list = conv_configure(self.train_x_place, cnn_setting, num_classes, logger)

        batch_size, f_num, v_num, c_num = last_conv_out.get_shape().as_list()
        pool_row = f_num
        pool_col = 1
        print(last_conv_out.get_shape())
        last_conv_out = conf_pool_layer(last_conv_out, pool_row, pool_col)
        batch_size, f_num, v_num, c_num = last_conv_out.get_shape().as_list()
        #print(last_conv_out.get_shape())
        self.saver_file = self.saver_file + saver_file
        std_value = cnn_setting.std_value
        #self.keeped_feature_list.append(last_conv_out)
        full_feature_num = cnn_setting.full_feature_num
        self.dropout_place = tf.placeholder(tf.float32)
        logits_out = out_configure(layer, last_conv_out, num_classes, self.dropout_place, full_feature_num, std_value, logger)
        return logits_out
    #return train_x_placeholder, output_y_placeholder, logits_out, keep_prob_placeholder, keeped_feature_list, saver_file

    def cnn_train_init(self, logits_out):
        cnn_setting = self.cnn_setting
        self.predict_y_proba = tf.nn.softmax(logits_out)
        #sess = tf.Session(config=tf.Config(allow_soft_placement=True, log_device_placement=True))
        self.cnn_session = tf.Session(config=self.config)
        cross_entropy, self.coefficient_place = cross_entropy_setup(self.cnn_session, self.data_stru.num_classes, logits_out, self.train_y_place, cnn_setting.l2_bool)
        eval_method_value, eval_method_key = eval_method_setup(self.train_y_place, self.predict_y_proba, self.cnn_setting.eval_method, self.data_stru.num_classes)

        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)
        self.logger.info("learning rate: " + str(self.learning_rate))
        #self.train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
        #self.cnn_session = tf.InteractiveSession()
        self.cnn_session.run(tf.global_variables_initializer())
        self.cross_entropy = cross_entropy
        self.data_group.data_check(self.data_stru.num_classes, self.data_stru.min_class)
        return eval_method_value, eval_method_key


    def cnn_training(self, eval_method_value, eval_method_key, up_train_step=None):
        logger = self.logger
        cnn_setting = self.cnn_setting
        cross_entropy_type = cnn_setting.cross_entropy_type
        logger.info("cnn setting inside cnn_training")
        logger.info(cnn_setting.to_string())
        data_stru = self.data_stru
        num_classes = data_stru.num_classes

        train_x_matrix = self.data_group.train_x_matrix
        train_y_matrix = self.data_group.train_y_matrix
        train_y_vector = self.data_group.train_y_vector
        valid_x_matrix = self.data_group.valid_x_matrix
        valid_y_matrix = self.data_group.valid_y_matrix

        train_x_place = self.train_x_place
        train_y_place = self.train_y_place
        dropout_place = self.dropout_place
        coefficient_place = self.coefficient_place
        is_train_place = self.is_train_place

        if up_train_step is None:
            train_step = self.train_step
        else:
            train_step = up_train_step
        cnn_session = self.cnn_session
        saver_file = self.saver_file
        best_saver_file = saver_file.replace(".ckpt", "_best.ckpt")

        predict_y_proba = self.predict_y_proba

        eval_method = cnn_setting.eval_method
        batch_size = cnn_setting.batch_size
        stop_threshold = cnn_setting.stop_threshold
        max_iter = cnn_setting.max_iter
        i = 0
        start = 0
        epoch = 0
        end = batch_size
        batch_each_class = int(batch_size/num_classes)
        if batch_each_class < 3:
            batch_each_class = 3
        overall_len = data_stru.train_ins
        train_class_index_dict, train_min_length, train_max_length = class_label_vector_checking(train_y_vector)

        saver = tf.train.Saver()
        train_run_time = 0
        epoch_train_time = []
        np.random.seed(epoch)
        batch_index = np.random.permutation(overall_len)
        logger.info("Random Epoch: " + str(epoch) + str(batch_index[0:5]))
        keep_prob_val = cnn_setting.keep_prob_val
        valid_eval_value = -1
        best_eval_value = -1
        # attention type 0: Apply attented cross-entropy
        if cross_entropy_type == 0:
            attention_weight = np.ones((num_classes))
            no_acc_list = np.zeros((num_classes))
            attention_apply = False
        while(valid_eval_value < stop_threshold):
            if start >= overall_len:
                start = 0
                end = start + batch_size
                epoch = epoch + 1
                np.random.seed(epoch)
                if epoch % 100 ==0:
                    logger.info("Random Epoch: " + str(epoch) + str(batch_index[0:5]))
                    logger.info(str(epoch) + " epoch trianing time: " + str(train_run_time))
                    epoch_train_time.append(train_run_time)
                batch_index = np.random.permutation(overall_len)
                if cross_entropy_type == 0:
                    if max(no_acc_list) > 0:
                        attention_weight = 1 + no_acc_list/max(no_acc_list)
                        attention_apply = True
                        no_acc_list = np.zeros((num_classes))
            elif end > overall_len:
                end = overall_len

            batch_x_matrix = train_x_matrix[batch_index[start:end], :, :, :]
            batch_y_matrix = train_y_matrix[batch_index[start:end], :]
            if eval_method == "acc" or eval_method == 'f1':
                if i == 0:
                    logger.info("Batch controlled")
                batch_x_matrix, batch_y_matrix, coefficients_vector = batch_control(batch_x_matrix, batch_y_matrix, self, i, train_class_index_dict, batch_each_class)
                batch_max_len = float(max(coefficients_vector))
                coefficients_vector = batch_max_len/coefficients_vector
                if cross_entropy_type == 0 and epoch > 0:
                    batch_pred_prob = cnn_session.run(predict_y_proba, feed_dict={train_x_place: batch_x_matrix, dropout_place: 1.0, is_train_place: False})
                    batch_pred_vector = np.argmax(batch_pred_prob, axis=1)
                    if len(batch_y_matrix.shape) == 2:
                        batch_y_vector = np.argmax(batch_y_matrix, axis=1)
                    elif len(batch_y_matrix.shape) == 3:
                        batch_y_vector = np.argmax(batch_y_matrix[:, :, 1], axis=1)
                    class_acc_list = class_based_accuracy(batch_pred_vector, batch_y_vector)
                    no_acc_list = no_acc_list + (1 - class_acc_list)
                    if i % 100 == 0:
                        logger.info("Attention at epoch " +str(epoch))
                        logger.info("accuracy list")
                        logger.info(class_acc_list)
                        logger.info("Non acc list")
                        logger.info(no_acc_list)
                        logger.info("atten weight")
                        logger.info(attention_weight)

                    if attention_apply is True:
                        if i % 100 == 0:
                            logger.info("before attention weight")
                            logger.info(coefficients_vector)
                        coefficients_vector = np.multiply(coefficients_vector, attention_weight)
                        if i % 100 == 0:
                            logger.info("after attention weight")
                            logger.info(coefficients_vector)

                if i % 100 == 0:
                    logger.info("Weighted train")
                start_time = time.time()
                cnn_session.run(train_step, feed_dict={train_x_place: batch_x_matrix, train_y_place: batch_y_matrix, coefficient_place: coefficients_vector, dropout_place: keep_prob_val, is_train_place: True})
                #train_step.run(feed_dict={train_x_place: batch_x_matrix, train_y_place: batch_y_matrix, coefficient_place: coefficients_vector, dropout_place: keep_prob_val})
                train_run_time = train_run_time + time.time() - start_time
            else:
                if i % 100 == 0:
                    logger.info("unweighted train")
                start_time = time.time()
                #train_step.run(feed_dict={train_x_place: batch_x_matrix, train_y_place: batch_y_matrix, coefficient_place: coefficients_vector, dropout_place: keep_prob_val})
                cnn_session.run(train_step, feed_dict={train_x_place: batch_x_matrix, train_y_place: batch_y_matrix, coefficient_place: coefficients_vector, dropout_place: keep_prob_val, is_train_place: True})
                train_run_time = train_run_time + time.time() - start_time
            if i % 100 == 0:
                if valid_x_matrix is not None:
                    #valid_eval_value = eval_method_value.eval(feed_dict={train_x_place: valid_x_matrix, train_y_place: valid_y_matrix, dropout_place: 1.0})
                    valid_eval_value = cnn_session.run(eval_method_value, feed_dict={train_x_place: valid_x_matrix, train_y_place: valid_y_matrix, dropout_place: 1.0, is_train_place: False})
                    #cnn_predict_proba = cnn_session.run(predict_y_proba, feed_dict={train_x_place:  valid_x_matrix, dropout_place: 1.0})
                    #logger.info("cnn predict proba")
                    #logger.info(cnn_predict_proba[0:2, :])
                    #cnn_pred_y_vector = np.argmax(cnn_predict_proba, axis=1)
                    #cnn_acc_value = accuracy_score(valid_y_vector, cnn_pred_y_vector, True)
                    if str(valid_eval_value) == 'nan':
                        valid_eval_value = 0
                    print_str = "step " + str(i) + ", testing " + eval_method_key + ": " + str(valid_eval_value)
                    logger.info(print_str)
                    #logger.info("accuracy score: " + str(cnn_acc_value))
                    if best_eval_value < valid_eval_value:
                        best_eval_value = valid_eval_value
                        save_path = saver.save(cnn_session, best_saver_file)
                        print_str = "validation eval value at current epoch: " + str(best_eval_value)
                        logger.info(print_str)
        
            i = i + 1
            start = end
            end = end + batch_size
            if epoch > max_iter:
                logger.info("validation eval value at epoch: " + str(epoch))
                break

        if valid_x_matrix is not None:
            #valid_eval_value = eval_method_value.eval(feed_dict={train_x_place: valid_x_matrix, train_y_place: valid_y_matrix, dropout_place: 1.0})
            valid_eval_value = cnn_session.run(eval_method_value, feed_dict={train_x_place: valid_x_matrix, train_y_place: valid_y_matrix, dropout_place: 1.0, is_train_place: False})
            if valid_eval_value < best_eval_value:
                cnn_session.close()
                cnn_session = tf.Session(config=self.config)
                saver.restore(cnn_session, best_saver_file)
                self.cnn_session = cnn_session
            else:
                best_eval_value = valid_eval_value
                save_path = saver.save(cnn_session, best_saver_file)
                logger.info(print_str)
            if best_eval_value < valid_eval_value:
                best_eval_value = valid_eval_value
            logger.info("Running iteration: %d" % (i))
            #logger.info("final best " + eval_method_key + ": " + str(best_eval_value))
            #logger.info("final valid before" + eval_method_key + ": " + str(valid_eval_value))
            #valid_eval_value = cnn_session.run(eval_method_value, feed_dict={train_x_place: valid_x_matrix, train_y_place: valid_y_matrix, dropout_place: 1.0, is_train_place: False})
            #logger.info("final valid after" + eval_method_key + ": " + str(valid_eval_value))
        logger.info("Epoch training time list: " + str(epoch_train_time))
        self.cnn_session = cnn_session
        return train_run_time

    def cnn_train_main(self):
        cnn_setting = self.cnn_setting
        logger = self.logger
        logger.info(cnn_setting)
        input_map = cnn_setting.input_map

        logits_out = self.cnn_graph_setup(self.data_stru, cnn_setting, input_map)
        eval_method_value, eval_method_key = self.cnn_train_init(logits_out)
        # Even without any data, this class still can get here and create the graph without training
        # After here, the training requires at least both the train_x_matrix and train_y_vector
        # Reutrn the training time
        return self.cnn_training(eval_method_value, eval_method_key), eval_method_value

    def cnn_pred_main(self, eval_method_value):
        test_x_matrix = self.data_group.test_x_matrix
        test_y_matrix = self.data_group.test_y_matrix
        if test_x_matrix is None:
            return False
        start_time = time.time()
        eval_value = self.cnn_session.run(eval_method_value, feed_dict={self.train_x_place: test_x_matrix, self.train_y_place: test_y_matrix, self.dropout_place: 1.0, self.is_train_place: False})
        test_run_time = time.time() - start_time
        return eval_value, test_run_time


def conf_act(input_conv, activation_fun=0, logger=None):
    logger.info("act: " + str(activation_fun))
    if activation_fun == 0:
        eval_type = 0
        ret_shape = input_conv.get_shape()
        attr_len = int(ret_shape[1])
        if len(ret_shape) != 4 or attr_len <= 1:
            ret_conv = tf.nn.relu(input_conv)
        else:
            ret_conv = conf_topk_feature(input_conv, logger, eval_type)
            ret_conv = tf.nn.relu(ret_conv)
    elif activation_fun == 2:
        # Only keep the top 90% of kernels
        logger.info("act: " + str(activation_fun))
        ret_conv = tf.nn.relu(input_conv)
        top_k = -1
        ret_conv = conf_topk_kernel(ret_conv, top_k, logger, activation_fun)
    elif activation_fun == 3:
        logger.info("act3: " + str(activation_fun))
        ret_conv = tf.nn.relu(input_conv)

    elif activation_fun == 4:
        ret_shape = input_conv.get_shape()
        eval_type = 1
        attr_len = int(ret_shape[1])
        if len(ret_shape) != 4 or attr_len <= 1:
            ret_conv = tf.nn.relu(input_conv)
        else:
            ret_conv = conf_topk_feature(input_conv, logger, eval_type)
            ret_conv = tf.nn.relu(ret_conv)
    elif activation_fun == 5:
        eval_type = 0
        logger.info("act: " + str(activation_fun))
        ret_shape = input_conv.get_shape()
        eval_type = 1
        attr_len = int(ret_shape[1])
        if len(ret_shape) != 4 or attr_len <= 1:
            ret_conv = tf.nn.relu(input_conv)
        else:
            ret_conv = conf_topk_feature(input_conv, logger, eval_type)
            #ret_conv = tf.nn.relu(ret_conv)
    elif activation_fun == -1:
        eval_type = 0
        logger.info("act0: " + str(activation_fun))
        ret_shape = input_conv.get_shape()
        attr_len = int(ret_shape[1])
        if len(ret_shape) != 4 or attr_len <= 1:
            ret_conv = tf.nn.relu(input_conv)
        else:
            ret_conv = conf_feature_reorder(input_conv, logger, eval_type)
            ret_conv = tf.nn.relu(input_conv)
    
    return ret_conv

def conf_conv_layer(layer, kernel_r, kernel_c, input_matrix, num_input_map, num_output_map, is_train, activation_fun=0, strides_list=[1, 1, 1, 1], std_value=0.02, same_size=False, logger=None):
    #with tf.variable_scope("conv"):
    weight_variable = tf.Variable(tf.truncated_normal([kernel_r, kernel_c, num_input_map, num_output_map], stddev=std_value, seed=layer), name='conv_w_'+str(layer))

    #weight_variable = tf.get_variable('conv_w_'+str(layer), [kernel_r, kernel_c, num_input_map, num_output_map], initializer=tf.contrib.layers.xavier_initializer())
    #sn_iters = 1
    #update_collection = None
    #weight_variable = spectral_normed_weight(weight_variable, num_iters=sn_iters, update_collection=update_collection, name='conv_w_'+str(layer))
    
    
    bias_variable = tf.get_variable('conv_b_'+str(layer), [num_output_map], initializer=tf.zeros_initializer())
    #bias_variable = tf.Variable(tf.constant(0.0, shape=[num_output_map]), name='conv_b_'+str(layer))
    if same_size == "True":
        str_padding = 'SAME'
    else:
        str_padding = 'VALID'

    ret_conv_before_act = tf.nn.conv2d(input_matrix, weight_variable, strides=[1, 1, 1, 1], padding=str_padding) + bias_variable

    #ret_conv_before_act = tf.contrib.layers.batch_norm(ret_conv_before_act, center=True, scale=True, is_training=is_train, scope='bn' +str(layer))

    ret_conv = conf_act(ret_conv_before_act, activation_fun, logger)
    return ret_conv




# bc_type: the type of batch controlled
# bc_type is 0: The proposed batch controlled method
# bc_type is 1: the normal batch weight
# rand_class_index_dict: contains the indexes for all the class instances
#def batch_control(batch_x_matrix, batch_y_matrix, train_x_matrix, train_y_matrix, iter_num, batch_each_class, min_class, max_class, train_class_index_dict, logger=None):
def batch_control(batch_x_matrix, batch_y_matrix, cnn_model, iter_num, train_class_index_dict, batch_each_class=100):
    logger = cnn_model.logger
    data_group = cnn_model.data_group
    train_x_matrix = data_group.train_x_matrix
    train_y_matrix = data_group.train_y_matrix
    min_class = cnn_model.data_stru.min_class
    num_classes = cnn_model.data_stru.num_classes
    # BATCH_CONTROLLED
    if len(batch_y_matrix.shape) == 2:
        batch_y_vector = np.argmax(batch_y_matrix, axis=1)
    elif len(batch_y_matrix.shape) == 3:
        batch_y_vector = np.argmax(batch_y_matrix[:, :, 1], axis=1)
    batch_class_index_dict, batch_min_length, batch_max_length = class_label_vector_checking(batch_y_vector)
    if iter_num < 101 and iter_num % 100 == 0:
        logger.info("class index before: ")
        logger.info(batch_class_index_dict)

    coefficients_vector = []
    batch_class_index_dict_keys = batch_class_index_dict.keys()
    for c_label in range(min_class, num_classes):
        if c_label not in batch_class_index_dict_keys:
            c_label_index = train_class_index_dict[c_label]
            c_label_index_len = len(c_label_index)
            add_index_vector_len = 0
            if c_label_index_len > batch_each_class:
                add_index_vector = np.random.choice(c_label_index_len, batch_each_class, replace=False)
                add_index_vector_len = len(add_index_vector)
                #batch_x_matrix = np.concatenate((batch_x_matrix, train_x_matrix[c_label_index[add_index_vector], :, :, :]), axis=0)
                batch_x_matrix = np.concatenate((batch_x_matrix, train_x_matrix[c_label_index[add_index_vector], :]), axis=0)
                batch_y_matrix = np.concatenate((batch_y_matrix, train_y_matrix[c_label_index[add_index_vector], :]), axis=0)
            else:
                batch_x_matrix = np.concatenate((batch_x_matrix, train_x_matrix[c_label_index, :, :, :]), axis=0)
                batch_y_matrix = np.concatenate((batch_y_matrix, train_y_matrix[c_label_index, :]), axis=0)
                add_index_vector_len = c_label_index_len
        else:
            batch_class_index = batch_class_index_dict[c_label]
            add_index_vector_len = len(batch_class_index)
            c_label_index = train_class_index_dict[c_label]
            c_label_index_len = len(c_label_index)
            if add_index_vector_len < batch_each_class:
                add_count = batch_each_class - add_index_vector_len
                if c_label_index_len > add_count:
                    add_index_vector = np.random.choice(c_label_index_len, add_count, replace=False)
                    add_index_vector_len = add_index_vector_len + len(add_index_vector)
                    batch_x_matrix = np.concatenate((batch_x_matrix, train_x_matrix[c_label_index[add_index_vector], :]), axis=0)
                    batch_y_matrix = np.concatenate((batch_y_matrix, train_y_matrix[c_label_index[add_index_vector], :]), axis=0)
                else:
                    batch_x_matrix = np.concatenate((batch_x_matrix, train_x_matrix[c_label_index, :]), axis=0)
                    batch_y_matrix = np.concatenate((batch_y_matrix, train_y_matrix[c_label_index, :]), axis=0)
                    add_index_vector_len = add_index_vector_len + c_label_index_len
            #elif add_index_vector_len > 2 * batch_each_class:
            #        remove_count = (add_index_vector_len - 2 * batch_each_class)
            #        remove_index_vector = np.random.choice(batch_class_index, remove_count, replace=False)
            #        add_index_vector_len = add_index_vector_len - len(remove_index_vector)
            #        batch_x_matrix = np.delete(batch_x_matrix, remove_index_vector, axis=0)
            #        batch_y_matrix = np.delete(batch_y_matrix, remove_index_vector, axis=0)
        coefficients_vector.append(float(add_index_vector_len))
    if len(batch_y_matrix.shape) == 2:
        batch_y_vector = np.argmax(batch_y_matrix, axis=1)
    elif len(batch_y_matrix.shape) == 3:
        batch_y_vector = np.argmax(batch_y_matrix[:, :, 1], axis=1)
    batch_class_index_dict, batch_min_length, batch_max_length = class_label_vector_checking(batch_y_vector)
    if iter_num < 101 and iter_num % 100 == 0:
        logger.info("class index after: ")
        logger.info(batch_class_index_dict)
        logger.info("coefficient vector: ")
        logger.info(coefficients_vector)
    coefficients_vector = np.array(coefficients_vector)
    return batch_x_matrix, batch_y_matrix, coefficients_vector


def weighted_softmax_crossentropy(train_y_place, logits_out, coefficient_placeholder):
    # deduce weights for batch samples based on their true label
    weights = tf.reduce_sum(coefficient_placeholder * train_y_place, axis=1)
    # compute your (unweighted) softmax cross entropy loss
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=train_y_place, logits=logits_out)
    # apply the weights, relying on broadcasting of the multiplication
    weighted_losses = unweighted_losses * weights
    # reduce the result to get your final loss
    cross_entropy = tf.reduce_mean(weighted_losses)
    return cross_entropy


def eval_method_setup(train_y_place, predict_y_proba, eval_method, num_classes):
    actual = tf.argmax(train_y_place, axis=1)
    prediction = tf.argmax(predict_y_proba, axis=1)
    correct_prediction = tf.equal(actual, prediction)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    if eval_method =='f1':
        if num_classes==2:
            TP = tf.count_nonzero(prediction * actual, dtype=tf.float32)
            FP = tf.count_nonzero(prediction * (actual - 1), dtype=tf.float32)
            FN = tf.count_nonzero((prediction - 1) * actual, dtype=tf.float32)
            precision = (TP) / (TP + FP)
            recall = (TP) / (TP + FN)
            f1 = (2 * precision * recall) / (precision + recall)
            eval_method_value = f1
            eval_method_keyword = "f1"
        else:
            eval_method_value = accuracy
            eval_method_keyword = "multi-f1 to acc batch"
    elif eval_method == "acc":
        eval_method_value = accuracy
        eval_method_keyword = "acc with batch"
    elif eval_method=="acc_no_batch":
        eval_method_value = accuracy
        eval_method_keyword = "acc no batch"
    return eval_method_value, eval_method_keyword


def cross_entropy_setup(cnn_session, num_classes, logits_out, train_y_placeholder, l2_bool=False):
    coefficient_place = tf.placeholder(tf.float32, shape=[num_classes])

    cross_entropy = weighted_softmax_crossentropy(train_y_placeholder, logits_out, coefficient_place)

    if l2_bool is True:
        beta = 0.001
        #full_weight = tf.get_default_graph().get_tensor_by_name("out_full_w:0")
        full_weight = cnn_session.graph.get_tensor_by_name("out_full_w:0")
        regularizers = tf.nn.l2_loss(full_weight)
        cross_entropy = tf.reduce_mean(cross_entropy + regularizers * beta)
    return cross_entropy, coefficient_place

def conf_pool_layer(input_matrix, row_d_samp_rate, col_samp_rate, same_size=False):
    if same_size is True:
        str_padding = 'SAME'
    else:
        str_padding = 'VALID'
    return tf.nn.max_pool(input_matrix, ksize=[1, row_d_samp_rate, col_samp_rate, 1], strides=[1, row_d_samp_rate, col_samp_rate, 1], padding=str_padding)


def conf_out_layer_1(layer, input_x_matrix, num_features, num_classes, std_value=0.02):
    output_weight = tf.Variable(tf.truncated_normal([num_features, num_classes], stddev=std_value, seed=layer), name="out_w")
    output_bias = tf.Variable(tf.constant(std_value, shape=[num_classes]), name="out_b")
    # This is the logits used in the cross entropy. The predict_y_prab should be tf.nn.softmax(logits_out) or sigmoid(logits_out)
    logits_out = tf.matmul(input_x_matrix, output_weight) + output_bias
    return logits_out


def conf_out_layer(layer, input_x_matrix, num_features, num_classes, std_value=0.02, update_collection=None, attention_type=-1):
    # print(input_x_matrix.get_shape())
    #if attention_type == 0:
    #    last_conv_out, v_attn = v_attn_layer(input_x_matrix, "final_attn_layer")
    #c_attn_name = "c_attn"
    #c_attn_layer(input_x_matrix, num_classes, c_attn_name, update_collection)
    output = snlinear(input_x_matrix, num_classes, update_collection=update_collection, name='d_sn_linear')
    return output


def conv_configure(train_x_placeholder, cnn_setting, num_classes, logger=None):
    # CNN Parameters
    conv_kernel_list = cnn_setting.conv_kernel_list
    pool_rate_list = cnn_setting.pool_rate_list
    feature_num_list = cnn_setting.feature_num_list
    activation_fun = cnn_setting.activation_fun
    std_value = cnn_setting.std_value
    same_size = cnn_setting.same_size
    conv_row_num = len(conv_kernel_list)
    saver_file = '_act' + str(activation_fun)
    attention_type = cnn_setting.attention_type
    num_input_map = cnn_setting.input_map
    strides_list = [1, 1, 1, 1]
    attn_list = []
    is_train = tf.placeholder(tf.bool, name="is_train")
    last_conv_out = train_x_placeholder
    for i in range(0, conv_row_num):
        logger.info('layer: ' + str(i) + " input:")
        logger.info(last_conv_out.get_shape())
        conv_row_kernel = conv_kernel_list[i, 0]
        conv_col_kernel = conv_kernel_list[i, 1]
        train_x_row = int(last_conv_out.get_shape()[1])
        train_x_col = int(last_conv_out.get_shape()[2])

        if conv_row_kernel < 0:
            conv_row_kernel = train_x_row
        elif conv_row_kernel > train_x_row:
            conv_row_kernel = train_x_row

        num_output_map = feature_num_list[i]
        if conv_col_kernel > train_x_col:
            conv_col_kernel = train_x_col
        elif conv_col_kernel < 0:
            conv_col_kernel = train_x_col

        saver_file = saver_file + "_c" + str(conv_row_kernel) + "_" + str(conv_col_kernel)
        last_conv_out = conf_conv_layer(i, conv_row_kernel, conv_col_kernel, last_conv_out, num_input_map, num_output_map, is_train, activation_fun, strides_list, std_value, same_size, logger)
        logger.info("Conv output: " + str(last_conv_out.get_shape()))
        pool_row_kernel = pool_rate_list[i, 0]
        pool_col_kernel = pool_rate_list[i, 1]

        saver_file = saver_file + "_p" + str(pool_row_kernel) + "_" + str(pool_col_kernel)

        out_conv_shape = last_conv_out.get_shape()
        out_conv_row = int(out_conv_shape[1])
        out_conv_col = int(out_conv_shape[2])
        num_output_map = int(out_conv_shape[3])

        if pool_row_kernel > 0 and pool_col_kernel > 0:
            if pool_row_kernel > out_conv_row:
                warning_str = "Warning: given pooling row number " + str(pool_row_kernel) + \
                    " is bigger than the data row number " + str(out_conv_row)
                logger.info(warning_str)
                warning_str = "Setting the pooling row number to be the data row number"
                logger.info(warning_str)
                pool_row_kernel = out_conv_row
            if pool_col_kernel > out_conv_col:
                warning_str = "Warning: given pooling column number " + \
                    str(pool_col_kernel) + \
                    " is bigger than the data column number " + \
                    str(out_conv_row)
                logger.info(warning_str)
                warning_str = "Setting the pooling column number to be the data column number"
                logger.info(warning_str)
                pool_col_kernel = out_conv_col
            last_conv_out = conf_pool_layer(last_conv_out, pool_row_kernel, pool_col_kernel, same_size)
            logger.info("Pooling output type 1: " + str(last_conv_out.get_shape()))
        elif pool_row_kernel < 0 and pool_col_kernel < 0:
            last_conv_out = last_conv_out
        else:
            if pool_row_kernel < 0:
                pool_row_kernel = out_conv_row
            if pool_col_kernel < 0:
                pool_col_kernel = out_conv_col
            last_conv_out = conf_pool_layer(last_conv_out, pool_row_kernel, pool_col_kernel, same_size)
            logger.info("Pooling output type 3: " + str(last_conv_out.get_shape()))
        num_input_map = num_output_map

    batch_num, f_num, a_num, c_num = last_conv_out.get_shape().as_list()
    logger.info("Conv output" + str(last_conv_out.get_shape()))
    #if f_num > 100:
    #    pool_row = f_num/100
    #    pool_col = 1
    #    last_conv_out = conf_pool_layer(last_conv_out, pool_row, pool_col, False)
    #    logger.info("Conv shuffle output" + str(last_conv_out.get_shape()))
    all_f_num = f_num * a_num
    f_limit = 10000
    if all_f_num > f_limit:
        pool_row = all_f_num/f_limit
        pool_col = 1
        last_conv_out = conf_pool_layer(last_conv_out, pool_row, pool_col, False)
        logger.info("Conv shuffle output" + str(last_conv_out.get_shape()))
    if attention_type == 0:
        logger.info("Attention applied")
        logger.info("input shape: " + str(last_conv_out.get_shape()))
        #with tf.device("/cpu:0"):
        t_attn_out, t_attn = t_attn_layer(last_conv_out, "t_attn_layer")
        last_conv_out, v_attn = v_attn_layer(t_attn_out, "v_attn_layer")
        attn_list.append(t_attn)
        attn_list.append(v_attn)
        logger.info("Attention output" + str(last_conv_out.get_shape()))
    elif attention_type == 1:
        logger.info("Global Attention applied")
        sample_rate = int(all_f_num/2000)
        if sample_rate > 0:
            last_conv_out = tf.layers.max_pooling2d(inputs=last_conv_out, pool_size=[sample_rate, 1], strides=[sample_rate, 1])
        #with tf.device("/cpu:0"):
        last_conv_out, global_attn = global_attn_layer(last_conv_out, "global_attn_layer")
        logger.info("Global Attention output" + str(last_conv_out.get_shape()))
    elif attention_type == 2:
        logger.info("Input Attention applied")
        sample_rate = int(all_f_num/2000)
        if sample_rate > 0:
            last_conv_out = tf.layers.max_pooling2d(inputs=last_conv_out, pool_size=[sample_rate, 1], strides=[sample_rate, 1])
        last_conv_out = input_attn_layer(last_conv_out, "input_attn_layer")
        logger.info("Global Attention output" + str(last_conv_out.get_shape()))
    elif attention_type == 3:
        logger.info("Recurrent Attention with all channel")
        last_conv_out = input_attn_layer_with_all_channel(last_conv_out, "ra_all_channel")
        logger.info("Global Attention output" + str(last_conv_out.get_shape()))
    saver_file = saver_file + '.ckpt'
    layer = conv_row_num
    return last_conv_out, saver_file, layer, is_train, attn_list


def out_configure(layer, last_conv_out, num_classes, dropout_place, full_feature_num=400, std_value=0.02, logger=None):
    last_shape_len = len(last_conv_out.shape)
    if last_shape_len == 4:
        second_feature_num = int(last_conv_out.get_shape()[1] * last_conv_out.get_shape()[2] * last_conv_out.get_shape()[3])
        last_conv_out = tf.reshape(last_conv_out, [-1, second_feature_num])
    elif last_shape_len == 2:
        second_feature_num = int(last_conv_out.get_shape()[1])

    if full_feature_num > 0:
        weight_fullconn = tf.Variable(tf.truncated_normal([second_feature_num, full_feature_num], stddev=std_value, seed=layer), name="out_full_w")
        logger.info("full conn weight shape")
        logger.info(weight_fullconn.get_shape())
        bias_fullconn = tf.Variable(tf.constant(std_value, shape=[full_feature_num]), name="out_full_b")
    
        output_fullconn_no_act = tf.matmul(last_conv_out, weight_fullconn) + bias_fullconn
        output_fullconn = tf.nn.relu(output_fullconn_no_act)
        logger.info('last full connect layer output:')
        logger.info(str(output_fullconn.get_shape()))
    else:
        output_fullconn = last_conv_out
    #dropout
    output_fullconn_drop = tf.nn.dropout(output_fullconn, dropout_place)

    layer = layer + 1
    logits_out = conf_out_layer(layer, output_fullconn_drop, full_feature_num, num_classes, std_value)
    return logits_out



## CNN load and predict
# def cnn_set_flow_graph(data_stru, cnn_setting, input_map, logger=None):
#     attr_num = data_stru.attr_num
#     attr_len = data_stru.attr_len
#     num_classes = data_stru.num_classes
    
#     tf.reset_default_graph()
#     tf.random.set_random_seed(0)
#     output_y_placeholder = tf.placeholder(tf.float32, [None, num_classes])
#     train_x_placeholder = tf.placeholder(tf.float32, [None, attr_len, attr_num, input_map])
#     #logits_out, keep_prob_placeholder, keeped_feature_list, saver_file = cnn_configure(train_x_placeholder, cnn_setting, num_classes, logger)
#     last_conv_out, saver_file, layer, is_train, attn_list = conv_configure(train_x_placeholder, cnn_setting, num_classes, logger)
#     std_value = 0.02
#     keeped_feature_list = []
#     keeped_feature_list.append(last_conv_out)
#     full_feature_num = cnn_setting.full_feature_num
#     dropout_place = tf.placeholder(tf.float32)
#     logits_out = out_configure(layer, last_conv_out, num_classes, dropout_place, full_feature_num, std_value, logger)
#     return train_x_placeholder, output_y_placeholder, logits_out, dropout_place, keeped_feature_list, saver_file

def load_model(model_saved_file, data_stru, cnn_setting, logger=None):
    logger.info("load model function")
    logger.info(data_stru.attr_num)
    logger.info(data_stru.attr_len)
    input_map = cnn_setting.input_map
    logger.info(cnn_setting.to_string())

    train_x_placeholder, output_y_placeholder, logits_out, keep_prob_placeholder, keeped_feature_list, saver_file = cnn_set_flow_graph(data_stru, cnn_setting, input_map, logger)

    cnn_session = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(cnn_session, model_saved_file)
    return cnn_session, logits_out, train_x_placeholder, keep_prob_placeholder, keeped_feature_list



def load_model_predict(cnn_session, test_x_matrix, logits_out, train_x_placeholder, keep_prob_placeholder, is_train_place):
    cnn_predict_proba = cnn_session.run(logits_out, feed_dict={train_x_placeholder: test_x_matrix, keep_prob_placeholder: 1.0, is_train_place: False})
    return cnn_predict_proba

## End of CNN load and predict


def run_cnn(cnn_setting, data_group, saver_prefix="./", logger=None):
    cnn_model = cnn_model_class(data_group, cnn_setting, logger)
    cnn_model.saver_file = cnn_setting.out_model_folder + saver_prefix
    train_run_time, eval_method_value = cnn_model.cnn_train_main()
    eval_value, test_run_time = cnn_model.cnn_pred_main(eval_method_value)
    return eval_value, train_run_time, test_run_time, cnn_model

def main_test():
    train_ins = 500
    valid_ins = 5
    test_ins = 5
    attr_num = 45
    attr_len = 10
    input_map = 1
    num_classes = 4
    cross_entropy_type = 0

    train_x_matrix = np.random.rand(train_ins, attr_len, attr_num, input_map)
    train_y_vector = np.random.randint(num_classes, size=train_ins)
    test_x_matrix = np.random.rand(test_ins, attr_len, attr_num, input_map)
    test_y_vector = np.random.randint(num_classes, size=test_ins)
    valid_x_matrix = np.random.rand(valid_ins, attr_len, attr_num, input_map)
    valid_y_vector = np.random.randint(num_classes, size=valid_ins)
    print(train_x_matrix.shape)
    print(valid_x_matrix.shape)
    cnn_file = "../../parameters/cnn_model_parameter.txt"
    cnn_setting = return_cnn_setting_from_file(cnn_file)
    cnn_setting.l2_bool = True
    cnn_setting.cross_entropy_type = 0
    print(cnn_setting.to_string())
    class_column = 0
    data_group = data_collection(train_x_matrix, train_y_vector, class_column)
    data_group.valid_x_matrix = valid_x_matrix
    data_group.valid_y_vector = valid_y_vector
    data_group.test_x_matrix = test_x_matrix
    data_group.test_y_vector = test_y_vector
    data_stru = data_group.gene_data_stru()
    data_group.data_check(data_stru.num_classes, data_stru.min_class)
    pred_y_prob, train_run_time, test_run_time, cnn_model = run_cnn(cnn_setting, data_group, "./", None)

    last_conv_out = cnn_model.keeped_feature_list[0]
    train_last_conv = cnn_model.cnn_session.run(last_conv_out, feed_dict={cnn_model.train_x_place: data_group.train_x_matrix, cnn_model.dropout_place: 1.0, cnn_model.is_train_place: False})
    print(train_last_conv.shape)
    #print(pred_y_prob)


def class_label_vector_checking(y_vector):
    min_class = min(y_vector)
    max_class = max(y_vector)
    class_index_dict = {}
    min_length = -1
    max_length = -1
    for c in range(min_class, max_class+1):
        c_index = np.where(y_vector==c)[0]
        class_index_dict[c] = c_index
        if min_length == -1:
            min_length = len(c_index)
        elif len(c_index) < min_length:
            min_length = len(c_index)
        if max_length == -1:
            max_length = len(c_index)
        elif len(c_index) > max_length:
            max_length = len(c_index)

    return class_index_dict, min_length, max_length

if __name__ == '__main__':
    main_test()
