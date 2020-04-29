import tensorflow as tf
import numpy as np
import time
import sys
import os
from model_setting import nn_parameters
from model_cnn import cross_entropy_setup
from model_cnn import batch_control

sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'data_processing/'))
from data_processing import class_label_vector_checking

sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'fileio/'))
from log_io import setup_logger




# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
def run_nn(train_x_matrix, train_y_matrix, test_x_matrix, test_y_matrix, nn_setting, logger=None):
    if logger is None:
        logger = setup_logger('')

    x_row, x_col = train_x_matrix.shape
    y_row, y_col = train_y_matrix.shape
    num_classes = y_col
    train_x_placeholder, train_y_placeholder, logits_out, keep_prob_placeholder = configure_nn(x_col, num_classes, nn_setting)
    best_eval_value, train_run_time, test_run_time, nn_predict_proba = nn_train(train_x_matrix, train_y_matrix, test_x_matrix, test_y_matrix, train_x_placeholder, train_y_placeholder, logits_out, keep_prob_placeholder, nn_setting, logger)
    return best_eval_value, train_run_time, test_run_time, nn_predict_proba

##
def conf_nn_layers(train_x_col, input_placeholder, nn_setting, logger=None):
    if logger is None:
        logger = setup_logger('')
    layer_list = nn_setting.layer_list
    std_value = nn_setting.std_value
    layer_out = input_placeholder
    layer_iter = 0
    layer_input = train_x_col
    keep_prob_placeholder = tf.placeholder(tf.float32)
    for neurons in layer_list:
        weight_name = "weight_" + str(layer_iter)
        bias_name = "bias_" + str(layer_iter)
        weight = tf.Variable(tf.random_normal([layer_input, neurons], stddev=std_value, seed=layer_iter), name=weight_name)
        bias = tf.Variable(tf.zeros([neurons]), name=bias_name)
        layer_input = neurons
        hidden_out = tf.add(tf.matmul(layer_out, weight), bias)
        layer_out = tf.nn.relu(hidden_out)

        layer_out = tf.nn.dropout(layer_out, keep_prob_placeholder)
        layer_iter = layer_iter + 1
    return layer_out, layer_iter, keep_prob_placeholder


def conf_nn_out(input_matrix, num_classes, std_value, layer):
    layer_input = int(input_matrix.get_shape()[1])
    weight = tf.Variable(tf.random_normal([layer_input, num_classes], stddev=std_value, seed=layer), name="out_weight")
    bias = tf.Variable(tf.random_normal([num_classes], stddev=std_value, seed=layer), name="out_bias")
    return tf.add(tf.matmul(input_matrix, weight), bias)

# Both train_x and train_y are 2-d matrixes
def configure_nn(train_x_col, num_classes, nn_setting, logger=None):
    if logger is None:
        logger = setup_logger('')
    std_value = nn_setting.std_value
    tf.reset_default_graph()
    tf.random.set_random_seed(0)

    train_x_placeholder = tf.placeholder(tf.float32, [None, train_x_col])
    train_y_placeholder = tf.placeholder(tf.float32, [None, num_classes])

    layer_out_matrix, layer_iter, keep_prob_placeholder = conf_nn_layers(train_x_col, train_x_placeholder, nn_setting, logger)
    logits_out = conf_nn_out(layer_out_matrix, num_classes, std_value, layer_iter)
    return train_x_placeholder, train_y_placeholder, logits_out, keep_prob_placeholder

def nn_train(train_x_matrix, train_y_matrix, test_x_matrix, test_y_matrix, train_x_placeholder, train_y_placeholder, logits_out, keep_prob_placeholder, nn_setting, logger):
    if logger is None:
        logger = setup_logger('')
    (overall_len, x_col) = train_x_matrix.shape
    (y_row, num_classes) = train_y_matrix.shape
    predict_y_proba = tf.nn.softmax(logits_out)
    train_y_vector = np.argmax(train_y_matrix, axis=1)
    max_class = max(train_y_vector)
    min_class = min(train_y_vector)

    eval_method = nn_setting.eval_method
    batch_size = nn_setting.batch_size
    stop_threshold = nn_setting.stop_threshold
    max_iter = nn_setting.max_epoch
    saver_file = nn_setting.save_file

    cross_entropy, eval_method_value, eval_method_keyword, coefficient_placeholder = cross_entropy_setup(eval_method, num_classes, logits_out, train_y_placeholder)
    beta = 0.001
    full_weight = tf.get_default_graph().get_tensor_by_name("weight_0:0")
    regularizers = tf.nn.l2_loss(full_weight)
    cross_entropy = tf.reduce_mean(cross_entropy + regularizers * beta)
    train_class_index_dict, train_min_length, train_max_length = class_label_vector_checking(train_y_vector)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    nn_session = tf.InteractiveSession()
    nn_session.run(tf.global_variables_initializer())
    
    test_eval_value = 0
    best_eval_value = 0
    i = 0
    start = 0
    epoch = 0
    end = batch_size
    batch_each_class = int(batch_size/num_classes)
    saver = tf.train.Saver()
    train_run_time = 0
    np.random.seed(epoch)
    batch_index = np.random.permutation(overall_len)
    logger.info("Random Epoch: " + str(epoch) + str(batch_index[0:5]))
    keep_prob_val = 0.5
    while(test_eval_value < stop_threshold):
        if start >= overall_len:
            start = 0
            end = start + batch_size
            epoch = epoch + 1
            np.random.seed(epoch)
            logger.info("Random Epoch: " + str(epoch) + str(batch_index[0:5]))
            print("Random Epoch: " + str(epoch) + str(batch_index[0:5]))
            batch_index = np.random.permutation(overall_len)
        elif end > overall_len:
            end = overall_len
        batch_x_matrix = train_x_matrix[batch_index[start:end], :]
        batch_y_matrix = train_y_matrix[batch_index[start:end], :]

        if eval_method == 'f1' or eval_method == "acc":
            if i == 0:
                logger.info("Batch controlled")
                print("Batch controled")
            batch_x_matrix, batch_y_matrix, coefficients_vector = batch_control(batch_x_matrix, batch_y_matrix, train_x_matrix, train_y_matrix, i, batch_each_class, min_class, max_class, train_class_index_dict, logger)
            
            batch_max_len = float(max(coefficients_vector))
            coefficients_vector = batch_max_len/coefficients_vector

            start_time = time.time()
            train_step.run(feed_dict={train_x_placeholder: batch_x_matrix, train_y_placeholder: batch_y_matrix, coefficient_placeholder: coefficients_vector, keep_prob_placeholder: keep_prob_val})
            train_run_time = train_run_time + time.time() - start_time
        else:
            start_time = time.time()
            train_step.run(feed_dict={train_x_placeholder: batch_x_matrix, train_y_placeholder: batch_y_matrix, keep_prob_placeholder: keep_prob_val})
            train_run_time = train_run_time + time.time() - start_time
        if i % 100 == 0:
            test_eval_value = eval_method_value.eval(feed_dict={
                train_x_placeholder: test_x_matrix, train_y_placeholder: test_y_matrix, keep_prob_placeholder: 1.0})
            if str(test_eval_value) == 'nan':
                test_eval_value = 0
            #print_str = "step " + str(i) + ", training " + eval_method_keyword + ": " + str(train_eval_value)
            #logger.info(print_str)
            print_str = "step " + str(i) + ", testing " + eval_method_keyword + ": " + str(test_eval_value)
            logger.info(print_str)
            print(print_str)
            if best_eval_value < test_eval_value:
                # Save the variables to disk.
                best_eval_value = test_eval_value
                save_path = saver.save(nn_session, saver_file)
                print_str = "Model saved in file: " + save_path + ' at iteration: ' + str(i)
                logger.info(print_str)
        
        i = i + 1
        start = end
        end = end + batch_size
        if epoch > max_iter:
            logger.info("best eval value at epoch: " + str(epoch))
            logger.info("best eval value to break")
            logger.info(best_eval_value)
            break

    start_time = time.time()
    test_eval_value = eval_method_value.eval(feed_dict={train_x_placeholder: test_x_matrix, train_y_placeholder: test_y_matrix, keep_prob_placeholder: 1.0})
    test_run_time = time.time() - start_time
    if test_eval_value < best_eval_value:
        nn_session.close()
        nn_session = tf.InteractiveSession()
        saver.restore(nn_session, saver_file)
    else:
        best_eval_value = test_eval_value
    
    logger.info("Running iteration: %d" % (i))
    logger.info("final best " + eval_method_keyword + ": " + str(best_eval_value))
    logger.info("final test " + eval_method_keyword + ": " + str(test_eval_value))
    print("final best " + eval_method_keyword + ": " + str(best_eval_value))
    print("final test " + eval_method_keyword + ": " + str(test_eval_value))


    nn_predict_proba = nn_session.run(predict_y_proba, feed_dict={train_x_placeholder: test_x_matrix, keep_prob_placeholder: 1.0})
    logger.info("NN model saved: " + str(saver_file))
    nn_session.close()
    return best_eval_value, train_run_time, test_run_time, nn_predict_proba


def main():
    train_y_vector = np.array([1,1,2,2,2,2,2,2,2,2, 3, 3, 3])
    train_y_predict_vector = np.array([2,2,2,2,2,2,2,2,2,2, 2, 2,2])
    min_class = min(train_y_vector)
    max_class = max(train_y_vector)
    num_classes = max_class - min_class + 1
    train_class_index_dict, train_min_length, train_max_length = class_label_vector_checking(train_y_vector)
    correct_prediction = tf.equal(train_y_vector, train_y_predict_vector)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    avg_accuracy = tf.Variable(0.0, tf.float32)
    for c_label in range(min_class, max_class+1):
        c_label_index = train_class_index_dict[c_label]
        class_y_vector = train_y_vector[c_label_index]
        class_y_predict = train_y_predict_vector[tf.constant(c_label_index)]
        class_accuracy = tf.reduce_mean(tf.cast(tf.equal(class_y_vector, class_y_predict), tf.float32))
        avg_accuracy = avg_accuracy + class_accuracy

    avg_accuracy = (avg_accuracy)/num_classes
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print (sess.run(accuracy))
        print (sess.run(avg_accuracy))

def main_test():
    train_x_matrix = np.array([[1,1,1], [1,2,2], [1,2,3], [2,2,2], [3,3,3], [3,2,1], [1,1,1], [1,2,2], [1,2,3], [2,2,2], [3,3,3], [3,2,1], [1,1,1], [1,2,2], [1,2,3], [2,2,2], [3,3,3], [3,2,1], [1,1,1], [1,2,2], [1,2,3], [2,2,2], [3,3,3], [3,2,1]])
    train_y_matrix = np.array([[0,1],[1,0], [1,0], [0,1], [0, 1], [1, 0], [0,1],[1,0], [1,0], [0,1], [0, 1], [1, 0], [0,1],[1,0], [1,0], [0,1], [0, 1], [1, 0], [0,1],[1,0], [1,0], [0,1], [0, 1], [1, 0]])
    test_x_matrix = np.array([[1,1,1], [1,2,2], [1,2,3], [2,2,2], [3,3,3], [3,2,1]])
    test_y_matrix = np.array([[0,1],[1,0], [1,0], [0,1], [0, 1], [1, 0]])

    #train_x_matrix = mnist.train.images
    #train_y_matrix = mnist.train.labels
    #test_x_matrix = mnist.test.images
    #test_y_matrix = mnist.test.labels
    print (train_x_matrix.shape)
    print (train_y_matrix.shape)
    print (test_y_matrix.shape)
    #train_x_matrix = test_x_matrix
    #train_y_matrix = test_y_matrix
    print(train_x_matrix.shape)
    print(train_y_matrix.shape)
    layer_list = np.array([300])
    num_classes = 2
    x_col = 3
    batch_size = 100
    learning_rate = 0.001
    logger = None
    saver_file = './test.save'
    nn_setting = nn_parameters(layer_list, batch_size, 5, 0.9, 3, 0.02, 'f1', saver_file)
    best_eval_value, train_run_time, test_run_time, nn_predict_proba = run_nn(train_x_matrix, train_y_matrix, test_x_matrix, test_y_matrix, nn_setting, logger)
    print(best_eval_value)
    print(train_run_time)
    print(test_run_time)
    print(nn_predict_proba)



if __name__ == '__main__':
    # run_simple_graph()
    # run_simple_graph_multiple()
    # simple_with_tensor_board()
    #nn_example()
    main_test()
