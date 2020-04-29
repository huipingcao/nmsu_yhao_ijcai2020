import sys
import numpy as np
from sklearn.metrics import accuracy_score

from fileio.data_io import train_test_file_reading, data_group_processing, list_files, init_folder
from fileio.log_io import init_logging
from fileio.parameter_proc import read_all_feature_classification

from tensor_model.model_cnn import run_cnn
from tensor_model.model_setting import return_cnn_setting_from_file

from utils.classification_results import averaged_class_based_accuracy

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'



# from classification_results import predict_matrix_with_prob_to_predict_accuracy
# from classification_results import f1_value_precision_recall_accuracy

# This is a multi-class classification using CNN model. Using Accuracy instead of F1 as measurement
# Just classification, no need to store the output objects
def cnn_classification_main(parameter_file, file_keyword, attention_type, function_keyword="fcn_classification"):
    data_keyword, data_folder, attr_num, attr_len, num_classes, start_class, class_column, class_id, obj_folder, method, log_folder, out_obj_folder, out_model_folder, cnn_setting_file, learning_rate = read_all_feature_classification(parameter_file, function_keyword)

    print(data_keyword, data_folder, attr_num, attr_len, num_classes, start_class, class_column, class_id, obj_folder, method, log_folder, out_obj_folder, out_model_folder, cnn_setting_file)

    log_folder = init_folder(log_folder)
    out_obj_folder = init_folder(out_obj_folder)
    out_model_folder = init_folder(out_model_folder)


    file_list = list_files(data_folder)
    file_count = 0

    class_column = 0
    header = True

    cnn_setting = return_cnn_setting_from_file(cnn_setting_file)
    conv_num = len(cnn_setting.conv_kernel_list)
    cnn_setting.out_obj_folder = out_obj_folder
    cnn_setting.out_model_folder = out_model_folder
    cnn_setting.learning_rate = learning_rate
    #cnn_setting.attention_type = 0     # 0: apply ra then sa attentions
                                        # -1: No attentions
    #cnn_setting.attention_type = -1
    #cnn_setting.attention_type = 1  # Using the global attention mechnizm
    #cnn_setting.attention_type = 2  # Using the input attention from https://arxiv.org/pdf/1704.02971.pdf
    
    cnn_setting.attention_type = attention_type
    cnn_setting.cross_entropy_type = -1   # 0: apply the class-based cross-entropy
                                          # -1: apply the normal cross-entropy
    #cnn_setting.cross_entropy_type = 0
    init_folder(out_obj_folder)
    init_folder(out_model_folder)
    
    result_obj_folder = obj_folder + method +"_result_folder"
    result_obj_folder = init_folder(result_obj_folder)

    delimiter = ' '
    loop_count = -1
    saver_file_profix = ""
    trans_bool = False  # True: means ins * attr_len * 1 * attr_num
                        # False: means ins * attr_len * attr_num * 1
    for train_file in file_list:
        if file_keyword not in train_file:
            continue
        loop_count = loop_count + 1
        file_key = train_file.replace('.txt', '')
        saver_file_profix = file_key + "_atten" + str(cnn_setting.attention_type)
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
        cnn_eval_key = cnn_setting.eval_method
        if cnn_eval_key == "f1":
            if num_classes > 2:
                cnn_eval_key = "acc"
        
        log_file = log_folder + data_keyword + '_' + file_key + '_' + function_keyword + "_act" + str(cnn_setting.activation_fun) + "_" + cnn_eval_key + "_attention" + str(cnn_setting.attention_type) + "_conv" + str(conv_num) + '.log'
    
        print("log file: " + log_file)
    
        logger = init_logging(log_file)
        logger.info('\nlog file: ' + log_file)
        logger.info(train_file)
        logger.info('cnn setting:\n ' + cnn_setting.to_string())
        logger.info('method: ' + method)
        logger.info('============')

        if file_count == 0:
            logger.info('train matrix shape: ' + str(data_group.train_x_matrix.shape))
            logger.info('train label shape: ' + str(data_group.train_y_vector.shape))

        logger.info(data_group.train_x_matrix[0, 0:3, 0:2, 0])
        eval_value, train_run_time, test_run_time, cnn_model = run_cnn(cnn_setting, data_group, saver_file_profix, logger)
        #pred_y_vector = np.argmax(pred_y_prob, axis=1)
        #avg_acc, ret_str = averaged_class_based_accuracy(pred_y_vector, data_group.test_y_vector)
        #acc_value = accuracy_score(data_group.test_y_vector, pred_y_vector, True)
        #logger.info("Averaged acc: " + str(acc_value))
        #logger.info(ret_str)
        logger.info("Fold accuracy: " + str(eval_value))
        logger.info(method + ' fold training time (sec):' + str(train_run_time))
        logger.info(method + ' fold testing time (sec):' + str(test_run_time))
        #logger.info("save obj to " + cnn_model.saver_file)


if __name__ == '__main__':
    argv_array = sys.argv
    run_stdout = sys.stdout
    file_keyword = 'train'
    projected = True
    len_argv_array = len(argv_array)
    if len_argv_array == 3:
        try:
            data_key = argv_array[1]
            attention_type = int(argv_array[2])
        except ValueError:
            print("That's not an int!")
    else:
        raise Exception("Unkonwn parameter detected! Please follow the format #python fcn_ca_main.py <DATA_NAME> <ATTENTION_TYPE>")
    print("dataset: " + data_key)
    print("attention type: " + str(attention_type))
    parameter_file = '../parameters/all_feature_classification_' + data_key.lower() + ".txt"
    print(parameter_file)
    cnn_classification_main(parameter_file, file_keyword, attention_type)
