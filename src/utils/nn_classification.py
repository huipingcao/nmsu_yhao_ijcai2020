import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'fileio/'))
from data_io import train_test_file_reading_with_attrnum
from data_io import list_files
from data_io import init_folder
from log_io import setup_logger
from parameter_proc import read_all_feature_classification

sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'data_processing/'))
from data_processing import y_vector_to_matrix
from data_processing import return_data_stru
from data_processing import train_test_transpose

sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'tensor_model/'))
from model_nn import run_nn
from model_setting import return_nn_setting_from_file
from model_setting import return_nn_keyword


# from classification_results import predict_matrix_with_prob_to_predict_accuracy
# from classification_results import f1_value_precision_recall_accuracy

# This is a multi-class classification using nn model. Using Accuracy instead of F1 as measurement
# Just classification, no need to store the output objects
def nn_classification_main(parameter_file, file_keyword, function_keyword="nn_classification"):
    data_keyword, data_folder, attr_num, attr_len, num_classes, start_class, class_column, class_id, obj_folder, method, log_folder, out_obj_folder, out_model_folder, nn_setting_file = read_all_feature_classification(parameter_file, function_keyword)

    print(data_keyword, data_folder, attr_num, attr_len, num_classes, start_class, class_column, class_id, obj_folder, method, log_folder, out_obj_folder, out_model_folder, nn_setting_file)

    log_folder = init_folder(log_folder)
    out_obj_folder = init_folder(out_obj_folder)
    out_model_folder = init_folder(out_model_folder)
    
    data_stru = return_data_stru(num_classes, start_class, attr_num, attr_len, class_column)

    file_list = list_files(data_folder)
    file_count = 0

    class_column = 0
    header = True

    nn_setting_file = "../../parameters/nn_model_parameter.txt"
    nn_setting, nn_key = return_nn_setting_from_file(nn_setting_file)
    
    result_obj_folder = obj_folder + method +"_result_folder"
    result_obj_folder = init_folder(result_obj_folder)

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

        train_x_matrix, train_y_vector, test_x_matrix, test_y_vector, attr_num = train_test_file_reading_with_attrnum(
            data_folder + train_file, data_folder + test_file, class_column, delimiter, header)
        min_class = min(train_y_vector)
        max_class = max(train_y_vector)
        num_classes = max_class - min_class + 1
        if nn_setting.eval_method == "accuracy":
            nn_eval_key = "acc"
        elif num_classes > 2:
            nn_eval_key = "acc_batch"
        else:
            nn_eval_key = "f1"
        log_file = log_folder + data_keyword + '_' + file_key + '_' + function_keyword + '_class' + str(min_class)+"_" + str(max_class) + "_act" + str(nn_setting.activation_fun) + "_" + nn_eval_key + '.log'
    
        print("log file: " + log_file)
    
        logger = setup_logger(log_file, 'logger_' + str(loop_count))
        logger.info('\nlog file: ' + log_file)
        logger.info(train_file)
        logger.info('nn setting:\n ' + nn_setting.to_string())
        logger.info('method: ' + method)
        logger.info('============')
        #train_y_vector[50:80] = 1
        #test_y_vector[30:40] = 1

        if file_count == 0:
            logger.info('train matrix shape: ' + str(train_x_matrix.shape))
            logger.info('train label shape: ' + str(train_y_vector.shape))
            logger.info('test matrix shape: ' + str(test_x_matrix.shape))
            logger.info('test label shape: ' + str(test_y_vector.shape))

        logger.info(train_x_matrix[0, 0:3])
        logger.info(test_x_matrix[0, 0:3])

        train_y_matrix = y_vector_to_matrix(train_y_vector, num_classes)
        test_y_matrix = y_vector_to_matrix(test_y_vector, num_classes)

        feature_dict = None
        top_k = -1
        #model_save_file = file_key + '_count' + str(file_count) + '_' + method 
        nn_eval_value, train_run_time, test_run_time, nn_predict_proba = run_nn(train_x_matrix, train_y_matrix, test_x_matrix, test_y_matrix, nn_setting, logger)

        logger.info("Fold eval value: " + str(nn_eval_value))
        logger.info(method + ' fold training time (sec):' + str(train_run_time))
        logger.info(method + ' fold testing time (sec):' + str(test_run_time))

if __name__ == '__main__':
    argv_array = sys.argv
    run_stdout = sys.stdout
    file_keyword = 'train_'
    projected = True
    len_argv_array = len(argv_array)
    if len_argv_array > 1:
        try:
            val = int(argv_array[1])
            file_keyword = file_keyword + argv_array[1]
        except ValueError:
           print("That's not an int!")

    parameter_file = '../../parameters/all_feature_classification.txt'
    nn_classification_main(parameter_file, file_keyword)
    #
