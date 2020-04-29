import os
import sys
import tensorflow as tf
import numpy as np

sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'fileio/'))
from data_io import train_test_file_reading
from data_io import list_files
from data_io import init_folder
from log_io import setup_logger
from parameter_proc import read_all_feature_classification

sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'data_processing/'))
from data_processing import y_vector_to_matrix
from data_processing import return_data_stru
from data_processing import train_test_transpose
from plot import plot_2dmatrix
from sklearn.metrics import accuracy_score

sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'tensor_model/'))
from model_cnn import run_cnn
from model_setting import return_cnn_setting_from_file
from model_setting import return_cnn_keyword
from model_setting import nn_parameters
from model_nn import run_nn
from model_cnn import load_model

from classification_results import averaged_class_based_accuracy


# predict_y_prob shape: num_classes * num_ins
# This is the only different with the result_analysis in classification_result
def load_result_analysis(predict_y_proba, test_y_vector):
    test_len = len(test_y_vector)
    predict_y_vector = np.argmax(predict_y_proba, axis=0)
    min_class = min(test_y_vector)
    max_class = max(test_y_vector) + 1
    for c in range(min_class, max_class):
        print("Class: " + str(c))
        for i in range(test_len):
            real_c = test_y_vector[i]
            pred_c = predict_y_vector[i]
            if real_c == c:
                if real_c != pred_c:
                    print("ins: " + str(i))
                    print("True label: " + str(real_c))
                    print("Predict label: " + str(pred_c))
                    print(predict_y_proba[:, i])




def conv_drop_index_gene(conv_weight, drop_ratio=0.1):
    weight_shape = conv_weight.shape
    print(np.squeeze(conv_weight).T[0:2, :])
    print(conv_weight.shape)
    #conv_weight = np.diff(conv_weight, axis=0)
    print(conv_weight.shape)
    print(np.squeeze(conv_weight).T[0:2, :])
    kernel_num = int(weight_shape[3])
    mean_row = np.mean(conv_weight, axis=-1)
    print(mean_row.shape)
    dist_list = []
    for r in range(kernel_num):
        row = conv_weight[:, :, :, r]
        dist_list.append(np.linalg.norm(row-mean_row))
    dist_list = np.array(dist_list)
    drop_count = int(kernel_num * drop_ratio)
    print(dist_list)
    #print(sorted(dist_list))
    print(dist_list.argsort())
    sdf
    drop_count = 10
    drop_index = dist_list.argsort()[-drop_count:][::-1]
    return drop_index

# input is numpy array
def conv_variable_up(conv_weight, conv_bias, drop_index):
    conv_weight[:, :, :, drop_index] = 0
    conv_bias[drop_index] = 0
    return conv_weight, conv_bias


def conv_variable_up_main(cnn_session, conv_count=3, drop_ratio=0.1):
    for conv_i in range(conv_count):
        weight_name = "conv_w_" + str(conv_i) + ":0"
        bias_name = "conv_b_" + str(conv_i) + ":0"
        weight_variable = tf.get_default_graph().get_tensor_by_name(weight_name)
        bias_variable = tf.get_default_graph().get_tensor_by_name(bias_name)

        weight_variable_val = cnn_session.run(weight_variable)
        bias_variable_val = cnn_session.run(bias_variable)
        drop_index = conv_drop_index_gene(weight_variable_val, drop_ratio)
        up_fir_weight, up_fir_bias = conv_variable_up(weight_variable_val, bias_variable_val, drop_index)
        weight_assign = tf.assign(weight_variable, up_fir_weight)
        bias_assign = tf.assign(bias_variable, up_fir_bias)
        cnn_session.run(weight_assign)
        cnn_session.run(bias_assign)
        #weight_variable_val = cnn_session.run(weight_variable)
        #bias_variable_val = cnn_session.run(bias_variable)
        #print(up_fir_bias)
        #print("second")
        #print(bias_variable_val)
        #print(weight_variable_val)



def last_conv_analysis(last_conv_out, y_vector):
    last_conv_out = np.squeeze(last_conv_out)
    min_class = min(y_vector)
    max_class = max(y_vector) + 1
    all_ins = range(len(last_conv_out))
    kernel_eval_matrix = []
    ref_kernel_eval_matrix = []
    for i in range(min_class, max_class):
        print("class: " + str(i))
        index_class = np.where(y_vector==i)[0]
        index_not_class = [x for x in all_ins if x not in index_class]

        class_last_out = last_conv_out[index_class, :]
        out_not_class = last_conv_out[index_not_class, :]
        kernel_dist = []
        ref_kernel_dist = []

        for k in range(20):
            print("kernel: " + str(k))
            kernel_class = class_last_out[:, k]
            kernel_not_class = out_not_class[:, k]
            #print(kernel_class.shape)
            kernel_class = np.mean(kernel_class, axis=0)
            print(kernel_class)
            kernel_not_class = np.mean(kernel_not_class, axis=0)
            print(kernel_not_class)
            print(kernel_class.argsort()[::-1])
            print(kernel_not_class.argsort()[::-1])
            #print(np.linalg.norm(kernel_class-kernel_not_class))
            
            # 1. Educlien distance
            # kernel_dist.append(np.linalg.norm(kernel_class-kernel_not_class))
            # 2. order Distance
            class_sorted = np.array(kernel_class.argsort())
            not_class_sorted = np.array(kernel_not_class.argsort())
            class_sorted_squ = class_sorted.argsort()
            not_class_sorted_squ = not_class_sorted.argsort()
            order_dist = np.sum(np.absolute(class_sorted_squ-not_class_sorted_squ))
            print(order_dist)
            ref_dist = np.linalg.norm(kernel_class-kernel_not_class)
            ref_kernel_dist.append(ref_dist)
            print(ref_dist)
            kernel_dist.append(order_dist)
        

        kernel_dist = np.array(kernel_dist)
        kernel_dist = np.array(kernel_dist.argsort()[::-1])
        ref_kernel_dist = np.array(ref_kernel_dist)
        ref_kernel_dist = np.array(ref_kernel_dist.argsort()[::-1])
        kernel_eval_matrix.append(kernel_dist)
        ref_kernel_eval_matrix.append(ref_kernel_dist)
    return np.array(kernel_eval_matrix), np.array(ref_kernel_eval_matrix)



def top_attr_x_matrix(last_conv_out, drop_num=30):
    x_ins, x_feature, x_attr, x_kernel = last_conv_out.shape
    for i in range(x_ins):
        for k in range(x_kernel):
            temp_matrix = np.squeeze(last_conv_out[i, :, :, k])
            temp_sorted = np.argsort(temp_matrix)
            last_conv_out[i, 0, temp_sorted[0:drop_num], k] = 0
    return last_conv_out
            




# from classification_results import predict_matrix_with_prob_to_predict_accuracy
# from classification_results import f1_value_precision_recall_accuracy

# This is a multi-class classification using CNN model. Using Accuracy instead of F1 as measurement
# Just classification, no need to store the output objects
def cnn_load_main(parameter_file, file_keyword, function_keyword="cnn_classification"):
    data_keyword, data_folder, attr_num, attr_len, num_classes, start_class, class_column, class_id, obj_folder, method, log_folder, out_obj_folder, out_model_folder, cnn_setting_file = read_all_feature_classification(parameter_file, function_keyword)

    print(data_keyword, data_folder, attr_num, attr_len, num_classes, start_class, class_column, class_id, obj_folder, method, log_folder, out_obj_folder, out_model_folder, cnn_setting_file)

    log_folder = init_folder(log_folder)
    out_obj_folder = init_folder(out_obj_folder)
    out_model_folder = init_folder(out_model_folder)
    
    data_stru = return_data_stru(num_classes, start_class, attr_num, attr_len, class_column)

    file_list = list_files(data_folder)
    file_count = 0

    class_column = 0
    header = True

    cnn_setting = return_cnn_setting_from_file(cnn_setting_file)
    cnn_setting.out_obj_folder = out_obj_folder
    cnn_setting.out_model_folder = out_model_folder
    cnn_setting.full_feature_num = 400
    init_folder(out_obj_folder)
    init_folder(out_model_folder)
    
    print (out_model_folder)
    model_file_list = list_files(out_model_folder)

    result_obj_folder = obj_folder + method +"_result_folder"
    result_obj_folder = init_folder(result_obj_folder)

    logger = setup_logger('')

    delimiter = ' '
    loop_count = -1
    saver_file_profix = ""
    for train_file in file_list:
        if file_keyword not in train_file:
            continue
        loop_count = loop_count + 1
        file_key = train_file.replace('.txt', '')
        saver_file_profix = file_key
        test_file = train_file.replace('train', 'test')

        #train_x_matrix, train_y_vector, test_x_matrix, test_y_vector, attr_num = train_test_file_reading(data_folder + train_file, data_folder + test_file, '', class_column, delimiter, header)
        data_group, attr_num = train_test_file_reading(data_folder + train_file, data_folder + test_file, '', class_column, delimiter, header)
        train_x_matrix = data_group.train_x_matrix
        train_y_vector = data_group.train_y_vector
        test_x_matrix = data_group.test_x_matrix
        test_y_vector = data_group.test_y_vector

        train_x_matrix = train_test_transpose(train_x_matrix, attr_num, attr_len, False)
        test_x_matrix = train_test_transpose(test_x_matrix, attr_num, attr_len, False)
        train_y_matrix = y_vector_to_matrix(train_y_vector, num_classes)
        test_y_matrix = y_vector_to_matrix(test_y_vector, num_classes)

        found_model_file = ""
        for model_file in model_file_list:
            if model_file.startswith(file_key):
                model_file = model_file.split('.')[0]
                found_model_file = out_model_folder + model_file + ".ckpt"
                break
        if found_model_file == "":
            raise Exception("No model object file found!!!")
        print(found_model_file)
        cnn_session, logits_out, train_x_placeholder, keep_prob_placeholder, keeped_feature_list = load_model(found_model_file, data_stru, cnn_setting, logger)

        last_conv_tensor = keeped_feature_list[0]
        train_last_conv = cnn_session.run(last_conv_tensor, feed_dict={train_x_placeholder: train_x_matrix, keep_prob_placeholder: 1.0})
        test_last_conv = cnn_session.run(last_conv_tensor, feed_dict={train_x_placeholder: test_x_matrix, keep_prob_placeholder: 1.0})
        drop_num = 10
        print(np.squeeze(test_last_conv[1, :, :, :]))
        test_last_conv = top_attr_x_matrix(test_last_conv, drop_num)
        print(np.squeeze(test_last_conv[1, :, :, :]))
        train_last_conv = top_attr_x_matrix(train_last_conv, drop_num)

        output_y_placeholder = tf.placeholder(tf.float32, [None, num_classes])
        actual = tf.argmax(output_y_placeholder, axis=1)
        prediction = tf.argmax(logits_out, axis=1)
        correct_prediction = tf.equal(actual, prediction)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        ori_pred_y_vector = cnn_session.run(prediction, feed_dict={train_x_placeholder: test_x_matrix, keep_prob_placeholder: 1.0})
        test_accuracy = cnn_session.run(accuracy, feed_dict={train_x_placeholder: test_x_matrix, keep_prob_placeholder: 1.0, output_y_placeholder: test_y_matrix})
        cnn_session.close()
        
        kernel_eval_matrix, ref_kernel_eval_matrix = last_conv_analysis(train_last_conv, train_y_vector)
        print(kernel_eval_matrix.shape)
        print(kernel_eval_matrix)
        train_ins_len = len(train_y_vector)
        test_ins_len = len(test_y_vector)
        batch_size = 100
        layer_list = np.array([400])
        max_epoch = 10
        stop_threshold = 0.99
        activation_fun = 3
        std_value = 0.02
        eval_method = "acc"
        saver_file = './test_1.save'
        nn_setting = nn_parameters(layer_list, batch_size, max_epoch, stop_threshold, activation_fun, std_value, eval_method, saver_file)
        all_pred_prob = []
        for c in range(num_classes):
            train_y_vector_class = np.zeros((train_ins_len))
            index_class = np.where(train_y_vector==c)[0]
            train_y_vector_class[index_class] = 1
            train_y_m_class = y_vector_to_matrix(train_y_vector_class, 2)

            test_y_vector_class = np.zeros((test_ins_len))
            index_class = np.where(test_y_vector==c)[0]
            test_y_vector_class[index_class] = 1
            test_y_m_class = y_vector_to_matrix(test_y_vector_class, 2)
            keep_num = 5
            kernel_index = kernel_eval_matrix[c, 0:keep_num]
            ref_kernel_index = ref_kernel_eval_matrix[c, 0:keep_num]
            print("kernel index " + str(kernel_index))
            print("ref kernel index " + str(ref_kernel_index))
            kernel_index = np.concatenate((kernel_index, ref_kernel_index), axis=0)
            print("union index " + str(kernel_index))
            kernel_index = np.unique(kernel_index)
            print("unique index " + str(kernel_index))

            kernel_index = ref_kernel_eval_matrix[c, 0:keep_num]
            train_x_class = train_last_conv[:, :, :, kernel_index]
            test_x_class = test_last_conv[:, :, :, kernel_index]
            print(train_x_class.shape)
            reshape_col = 45 * len(kernel_index)
            train_x_class = train_x_class.reshape((train_ins_len, reshape_col))
            test_x_class = test_x_class.reshape((test_ins_len, reshape_col))
            
            c_eval_value, c_train_time, c_test_time, c_predict_proba = run_nn(train_x_class, train_y_m_class, test_x_class, test_y_m_class, nn_setting)
            all_pred_prob.append(c_predict_proba[:, 1]-c_predict_proba[:, 0])
        all_pred_prob = np.array(all_pred_prob)
        print(all_pred_prob.shape)
        pred_vector = np.argmax(all_pred_prob, axis=0)
        print(pred_vector)
        print(all_pred_prob[:, 0])
        print(all_pred_prob[:, 1])
        print(all_pred_prob[:, 2])

        final_accuracy = accuracy_score(pred_vector, test_y_vector)

        avg_acc, ret_str = averaged_class_based_accuracy(ori_pred_y_vector, test_y_vector)
        print("original avg acc" + str(avg_acc))
        print("original accuracy: " + str(test_accuracy))
        print(ret_str)
        avg_acc, ret_str = averaged_class_based_accuracy(pred_vector, test_y_vector)
        print("avg acc" + str(avg_acc))
        print("new accuracy: " + str(final_accuracy))
        print(ret_str)

        load_result_analysis(all_pred_prob, test_y_vector)

        sdfds
        output_y_placeholder = tf.placeholder(tf.float32, [None, num_classes])
        actual = tf.argmax(output_y_placeholder, axis=1)
        prediction = tf.argmax(logits_out, axis=1)
        correct_prediction = tf.equal(actual, prediction)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        test_eval_value = accuracy.eval(feed_dict={train_x_placeholder: test_x_matrix, output_y_placeholder: test_y_matrix, keep_prob_placeholder: 1.0})
        print("fisrt")
        print(test_eval_value)

        


        conv_count = 1
        drop_ratio = 0.1

        #conv_variable_up_main(cnn_session, conv_count, drop_ratio)


        weight_name = "conv_w_" + str(0) + ":0"
        bias_name = "conv_b_" + str(0) + ":0"
        ori_weight_variable = tf.get_default_graph().get_tensor_by_name(weight_name)
        ori_bias_variable = tf.get_default_graph().get_tensor_by_name(bias_name)
        weight_variable = tf.get_default_graph().get_tensor_by_name(weight_name)
        bias_variable = tf.get_default_graph().get_tensor_by_name(bias_name)
        ori_weight_variable = cnn_session.run(weight_variable)
        ori_bias_variable = cnn_session.run(bias_variable)
        train_drop_acc = []
        test_drop_acc = []
        for drop_i in range(50):
            drop_weight_variable = np.copy(ori_weight_variable)
            drop_bias_variable = np.copy(ori_bias_variable)
            drop_index = []
            drop_index.append(drop_i)
            
            up_fir_weight, up_fir_bias = conv_variable_up(drop_weight_variable, drop_bias_variable, drop_index)
            weight_assign = tf.assign(weight_variable, up_fir_weight)
            bias_assign = tf.assign(bias_variable, up_fir_bias)
            cnn_session.run(weight_assign)
            cnn_session.run(bias_assign)
            up_bias_variable = tf.get_default_graph().get_tensor_by_name(bias_name)
            up_bias_variable_val = cnn_session.run(bias_variable)
            train_eval_value = accuracy.eval(feed_dict={train_x_placeholder: train_x_matrix, output_y_placeholder: train_y_matrix, keep_prob_placeholder: 1.0})
            train_drop_acc.append(train_eval_value)
            test_eval_value = accuracy.eval(feed_dict={train_x_placeholder: test_x_matrix, output_y_placeholder: test_y_matrix, keep_prob_placeholder: 1.0})
            test_drop_acc.append(test_eval_value)
            print ("Drop " + str(drop_i))
            print(train_eval_value)
            print(test_eval_value)
        
        print(train_drop_acc)
        print(train_drop_acc.argsort())
        print(test_drop_acc)
        print(test_drop_acc.argsort())

        sdfs
        print("HERE")



        fir_weight_variable_val = np.squeeze(fir_weight_variable_val)
        kernel_dist_val = cnn_session.run(kernel_dist)
        keep_index_val = cnn_session.run(keep_index)
        print(fir_weight_variable_val.shape)
        print(np.amax(fir_weight_variable_val, axis=1))
        print(np.amin(fir_weight_variable_val, axis=1))
        print(np.mean(fir_weight_variable_val, axis=1))
        mean_row = np.mean(fir_weight_variable_val, axis=-1)
        print(mean_row.shape)
        dist_list = []
        for r in range(40):
            row = fir_weight_variable_val[:, r]
            dist_list.append(np.linalg.norm(row-mean_row))
        print (dist_list)
        print(kernel_dist_val)
        print(keep_index_val)
        print(sorted(dist_list))
        print("!!!")
        #conv_variable_up(fir_weight_variable_val, fir_bias_variable_val)
        
        sdfsd

        train_x_matrix, train_y_vector, test_x_matrix, test_y_vector, attr_num = train_test_file_reading(data_folder + train_file, data_folder + test_file, class_column, delimiter, header)
        train_x_matrix = train_test_transpose(train_x_matrix, attr_num, attr_len, False)
        test_x_matrix = train_test_transpose(test_x_matrix, attr_num, attr_len, False)

        train_x_matrix = test_x_matrix[0:1, :, :, :]
        #plot_2dmatrix(np.squeeze(train_x_matrix)[:, 0:5])
        
        fir_out_tensor = tf.nn.conv2d(train_x_placeholder, fir_weight_variable, strides=[1, 1, 1, 1], padding='VALID') + fir_bias_variable
        fir_out_tensor = tf.nn.relu(fir_out_tensor)

        print(fir_out_tensor.get_shape())
        fir_analysis_tensor = tf.reduce_max(fir_out_tensor, [1])
        print(fir_analysis_tensor.get_shape())
        fir_analysis_tensor = tf.reduce_max(fir_analysis_tensor, [1])
        fir_analysis_tensor = tf.reduce_mean(fir_analysis_tensor, [0])

        top_k_indices = tf.nn.top_k(fir_analysis_tensor, 10).indices
        top_k_values = tf.nn.top_k(fir_analysis_tensor, 10).values
        top_fir_out_tensor = tf.gather(fir_out_tensor, top_k_indices, axis=3)

        sec_weight_variable = tf.get_default_graph().get_tensor_by_name("conv_w_1:0")
        sec_bias_variable = tf.get_default_graph().get_tensor_by_name("conv_b_1:0")
        sec_out_tensor = tf.nn.conv2d(fir_out_tensor, sec_weight_variable, strides=[1, 1, 1, 1], padding='VALID') + sec_bias_variable
        sec_out_tensor = tf.nn.relu(sec_out_tensor)
        sec_weight_var_val = cnn_session.run(sec_weight_variable)
        #print(np.squeeze(sec_weight_var_val))
        #sdfds

        #plot_2dmatrix(fir_weight_var_val[:, 4])
        #sdf
        #print(fir_weight_var_val.T)
        fir_out_tensor_val = cnn_session.run(fir_out_tensor, feed_dict={train_x_placeholder: train_x_matrix, keep_prob_placeholder: 1.0})
        print(fir_out_tensor_val.shape)

        top_fir_out_tensor = cnn_session.run(top_fir_out_tensor, feed_dict={train_x_placeholder: train_x_matrix, keep_prob_placeholder: 1.0})
        print(top_fir_out_tensor.shape)

        fir_analysis_tensor_val = cnn_session.run(fir_analysis_tensor, feed_dict={train_x_placeholder: train_x_matrix, keep_prob_placeholder: 1.0})
        print(fir_analysis_tensor.shape)

        top_k_indices_val = cnn_session.run(top_k_indices, feed_dict={train_x_placeholder: train_x_matrix, keep_prob_placeholder: 1.0})
        top_k_values_val = cnn_session.run(top_k_values, feed_dict={train_x_placeholder: train_x_matrix, keep_prob_placeholder: 1.0})
        fir_weight_variable_val = cnn_session.run(fir_weight_variable)
        fir_weight_variable_val = np.squeeze(fir_weight_variable_val)
        print(fir_weight_variable_val.shape)
        print(fir_analysis_tensor_val)
        fir_sort_in = np.argsort(fir_analysis_tensor_val)
        print(fir_sort_in)
        print(top_k_indices_val)
        print(top_k_values_val)
        plot_2dmatrix(fir_weight_variable_val[:, fir_sort_in[-10:]])
        sdfd





        for n in range(len(fir_out_tensor_val)):
            for k in range(50):
                ret_str = "k" + str(k) + ": "
                kernel_max = -1
                max_attr = -1
                max_attr_list = []
                for a in range(attr_num):
                    attr_max = max(fir_out_tensor_val[n, :, a, k])
                    max_attr_list.append(attr_max)
                    if attr_max > kernel_max:
                        kernel_max = attr_max
                        max_attr = a
                    if attr_max == 0:
                        ret_str = ret_str + str(a) + " "
                print(ret_str)
                print("max attr " + str(max_attr))
                print(sorted(range(len(max_attr_list)), key=lambda k: max_attr_list[k]))
                print("======")
        print("label " + str(train_y_vector[0]))
        fir_out_tensor_val = cnn_session.run(sec_out_tensor, feed_dict={train_x_placeholder: train_x_matrix, keep_prob_placeholder: 1.0})
        print(fir_out_tensor_val.shape)

        sdf

        for n in range(len(fir_out_tensor_val)):
            for k in range(40):
                ret_str = "k" + str(k) + ": "
                kernel_max = -1
                max_attr = -1
                max_attr_list = []
                for a in range(attr_num):
                    attr_max = max(fir_out_tensor_val[n, :, a, k])
                    max_attr_list.append(attr_max)
                    if attr_max > kernel_max:
                        kernel_max = attr_max
                        max_attr = a
                    if attr_max == 0:
                        ret_str = ret_str + str(a) + " "
                print(ret_str)
                print("max attr " + str(max_attr))
                print(sorted(range(len(max_attr_list)), key=lambda k: max_attr_list[k]))
                print("======")
        sdf

        fir_out_mean_val = cnn_session.run(fir_out_mean, feed_dict={train_x_placeholder: train_x_matrix, keep_prob_placeholder: 1.0})
        #fir_out_mean_val = np.squeeze(fir_out_mean_val)
        print(fir_out_mean_val.shape)

        plot_2dmatrix(np.squeeze(fir_out_mean_val[:, :, 0:5]))

        sdfd
        plot_2dmatrix(fir_weight_var_val)
        



        
        min_class = min(train_y_vector)
        max_class = max(train_y_vector)
        num_classes = max_class - min_class + 1
        if cnn_setting.eval_method == "accuracy":
            cnn_eval_key = "acc"
        elif num_classes > 2:
            cnn_eval_key = "acc_batch"
        else:
            cnn_eval_key = "f1"
        log_file = log_folder + data_keyword + '_' + file_key + '_' + function_keyword + '_class' + str(min_class)+"_" + str(max_class) + "_act" + str(cnn_setting.activation_fun) + "_" + cnn_eval_key + '.log'
    
        print("log file: " + log_file)
    
        logger = setup_logger(log_file, 'logger_' + str(loop_count))
        logger.info('\nlog file: ' + log_file)
        logger.info(train_file)
        logger.info('cnn setting:\n ' + cnn_setting.to_string())
        logger.info('method: ' + method)
        logger.info('============')

        train_x_matrix = train_test_transpose(train_x_matrix, attr_num, attr_len, False)
        test_x_matrix = train_test_transpose(test_x_matrix, attr_num, attr_len, False)
        if file_count == 0:
            logger.info('train matrix shape: ' + str(train_x_matrix.shape))
            logger.info('train label shape: ' + str(train_y_vector.shape))
            logger.info('test matrix shape: ' + str(test_x_matrix.shape))
            logger.info('test label shape: ' + str(test_y_vector.shape))

        logger.info(train_x_matrix[0, 0:3, 0:2, 0])
        logger.info(test_x_matrix[0, 0:3, 0:2, 0])

        train_y_matrix = y_vector_to_matrix(train_y_vector, num_classes)
        test_y_matrix = y_vector_to_matrix(test_y_vector, num_classes)










        cnn_eval_value, train_run_time, test_run_time, cnn_predict_proba, saver_file, feature_list_obj_file = run_cnn(train_x_matrix, train_y_matrix, test_x_matrix, test_y_matrix, data_stru, cnn_setting, saver_file_profix, logger)

        logger.info("Fold eval value: " + str(cnn_eval_value))
        logger.info(method + ' fold training time (sec):' + str(train_run_time))
        logger.info(method + ' fold testing time (sec):' + str(test_run_time))
        logger.info("save obj to " + saver_file)


if __name__ == '__main__':

    a = [33.302746, 0.32000482, 23.913399, 1.2215424, 65.93528, 10.179363, 1.9824872, 3.8264968, 48.5789, 49.488224, 1.1102881, 1.9492017, 56.94654, 7.652534, 47.50887, 61.650646, 22.168713, 5.890481, 56.289455, 3.5759919, ]
    b = [33.267616, 5.52648, 25.22736, 4.7585244, 60.202625, 12.542127, 5.895225, 6.737898, 43.182182, 45.2398, 4.7507195, 5.2565885, 51.47815, 10.358387, 43.13505, 55.500813, 23.074305, 8.803053, 51.40858, 6.9223094]
    a = np.array(a)
    b = np.array(b)
    a_sort = np.array(a.argsort())
    b_sort = np.array(b.argsort())
    print(a_sort)
    print(b_sort)

    a_sort_squ = np.array(a_sort.argsort())
    b_sort_squ = np.array(b_sort.argsort())
    print(a_sort_squ[0:4])
    print(b_sort_squ[0:4])
    print(np.linalg.norm(a_sort_squ - b_sort_squ))
    print(np.sum(np.absolute(a_sort_squ-b_sort_squ)))

    argv_array = sys.argv
    run_stdout = sys.stdout
    file_keyword = 'train_2'
    projected = True
    len_argv_array = len(argv_array)
    if len_argv_array > 1:
        try:
            val = int(argv_array[1])
            file_keyword = file_keyword + argv_array[1]
        except ValueError:
            print("That's not an int!")

    parameter_file = '../../parameters/all_feature_classification.txt'
    cnn_load_main(parameter_file, file_keyword)
    #
