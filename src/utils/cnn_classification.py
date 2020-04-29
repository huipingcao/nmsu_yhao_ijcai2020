import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score

sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'fileio/'))
from data_io import train_test_file_reading
from data_io import data_group_processing
from data_io import list_files
from data_io import init_folder
from log_io import setup_logger
from parameter_proc import read_all_feature_classification

sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'tensor_model/'))
from model_cnn import run_cnn
from model_setting import return_cnn_setting_from_file

from classification_results import averaged_class_based_accuracy


# from classification_results import predict_matrix_with_prob_to_predict_accuracy
# from classification_results import f1_value_precision_recall_accuracy

# This is a multi-class classification using CNN model. Using Accuracy instead of F1 as measurement
# Just classification, no need to store the output objects
def cnn_classification_main(parameter_file, file_keyword, function_keyword="cnn_classification"):
    data_keyword, data_folder, attr_num, attr_len, num_classes, start_class, class_column, class_id, obj_folder, method, log_folder, out_obj_folder, out_model_folder, cnn_setting_file = read_all_feature_classification(parameter_file, function_keyword)

    print(data_keyword, data_folder, attr_num, attr_len, num_classes, start_class, class_column, class_id, obj_folder, method, log_folder, out_obj_folder, out_model_folder, cnn_setting_file)

    log_folder = init_folder(log_folder)
    out_obj_folder = init_folder(out_obj_folder)
    out_model_folder = init_folder(out_model_folder)


    file_list = list_files(data_folder)
    file_count = 0

    class_column = 0
    header = True

    cnn_setting = return_cnn_setting_from_file(cnn_setting_file)
    cnn_setting.out_obj_folder = out_obj_folder
    cnn_setting.out_model_folder = out_model_folder
    init_folder(out_obj_folder)
    init_folder(out_model_folder)
    
    result_obj_folder = obj_folder + method +"_result_folder"
    result_obj_folder = init_folder(result_obj_folder)

    delimiter = ' '
    loop_count = -1
    saver_file_profix = ""
    attention_type = 0
    attention_type = -1
    cnn_setting.attention_type = attention_type
    trans_bool = False # True: means ins * attr_len * 1 * attr_num
                       # False: means ins * attr_len * attr_num * 1
    for train_file in file_list:
        if file_keyword not in train_file:
            continue
        loop_count = loop_count + 1
        file_key = train_file.replace('.txt', '')
        saver_file_profix = file_key + "_atten" + str(attention_type)
        valid_file = data_folder + train_file.replace('train', 'valid')
        if os.path.isfile(valid_file) is False:
            valid_file = ''

        test_file = data_folder + train_file.replace('train', 'test')
        if os.path.isfile(test_file) is False:
            test_file = ''

        data_group, attr_num = train_test_file_reading(data_folder + train_file, test_file, valid_file, class_column, delimiter, header)
        data_group_processing(data_group, attr_num, trans_bool)
        data_stru = data_group.gene_data_stru()
        data_group.data_check(data_stru.num_classes, data_stru.min_class)
        if cnn_setting.eval_method == "accuracy":
            cnn_eval_key = "acc"
        elif num_classes > 2:
            cnn_eval_key = "acc_batch"
        else:
            cnn_eval_key = "f1"
        log_file = log_folder + data_keyword + '_' + file_key + '_' + function_keyword + '_class' + str(data_stru.min_class)+"_" + str(data_stru.num_classes) + "_act" + str(cnn_setting.activation_fun) + "_" + cnn_eval_key + "_attention" + str(attention_type) + '.log'
    
        print("log file: " + log_file)
    
        logger = setup_logger(log_file, 'logger_' + str(loop_count))
        logger.info('\nlog file: ' + log_file)
        logger.info(train_file)
        logger.info('cnn setting:\n ' + cnn_setting.to_string())
        logger.info('method: ' + method)
        logger.info('============')

        if file_count == 0:
            logger.info('train matrix shape: ' + str(data_group.train_x_matrix.shape))
            logger.info('train label shape: ' + str(data_group.train_y_vector.shape))

        logger.info(data_group.train_x_matrix[0, 0:3, 0:2, 0])
        pred_y_prob, train_run_time, test_run_time, cnn_model = run_cnn(cnn_setting, data_group, saver_file_profix, logger)
        pred_y_vector = np.argmax(pred_y_prob, axis=1)
        avg_acc, ret_str = averaged_class_based_accuracy(pred_y_vector, data_group.test_y_vector)
        acc_value = accuracy_score(data_group.test_y_vector, pred_y_vector, True)
        logger.info("Averaged acc: " + str(acc_value))
        logger.info(ret_str)
        logger.info("Fold eval value: " + str(acc_value))
        logger.info(method + ' fold training time (sec):' + str(train_run_time))
        logger.info(method + ' fold testing time (sec):' + str(test_run_time))
        logger.info("save obj to " + cnn_model.saver_file)


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
    cnn_classification_main(parameter_file, file_keyword)
    #
