import logging
import numpy as np
from data_io import list_files
from data_io import init_folder
from data_io import write_to_excel
from object_io import save_obj
import os


def results_from_file(file_name, acc_keyword, train_time_key, test_time_key):
    feature_dict = {}
    min_class = 100
    max_class = -1
    acc_value = -1
    train_time = -1
    test_time = -1

    with open(file_name) as f:
        value_vector = []
        for line in f:
            if acc_keyword in line:
                line_array = line.split(':')
                acc_value = float(line_array[-1].strip())
            elif train_time_key in line:
                line_array = line.split(':')
                train_time = float(line_array[-1].strip())
            elif test_time_key in line:
                line_array = line.split(':')
                test_time = float(line_array[-1].strip())
            if acc_value != -1 and train_time != -1 and test_time != -1:
                break
    
    return acc_value, train_time, test_time


def results_from_folder(folder_name, file_keyword, acc_keyword, train_time_keyword, test_time_keyword, fold_count=10):
    file_list = list_files(folder_name)
    file_count = 0
    acc_list = []
    train_list = []
    test_list = []
    for fold_id in range(fold_count):
        fold_key = "train_" + str(fold_id) + "_"
        for file_name in file_list:
            if file_name.startswith('.'):
                continue
            if fold_key not in file_name:
                continue
            if file_keyword not in file_name:
                continue
            print(file_name)
            file_count = file_count + 1
            acc_value, train_time, test_time = results_from_file(folder_name+file_name, acc_keyword, train_time_keyword,    test_time_keyword)
            if len(acc_list) > fold_id:
                acc_list[fold_id] = acc_value
                train_list[fold_id] = train_time
                test_list[fold_id] = test_time
            else:
                acc_list.append(acc_value)
                train_list.append(train_time)
                test_list.append(test_time)
    print(acc_list)
    print(train_list)
    print(test_list)
        


if __name__ == '__main__':
    data_key = "dsa"
    data_key = "rar"
    #data_key = "arc"
    #data_key = "ara"
    #data_key = "asl"
    #data_key = "fixed_arc"
    #method = "cnn"
    #method = "knn"
    #method = "libsvm"
    #method = "rf"

    if data_key == "dsa":
        top_k = "_top15_"
        num_classes = 19
    elif data_key == "rar":
        top_k = "_top30_"
        num_classes = 33
    elif data_key == "arc" or data_key == "fixed_arc":
        top_k = "_top30_"
        num_classes = 18
    elif data_key == "ara":
        top_k = "_top4_"
        num_classes = 10
    elif data_key == "asl":
        top_k = "_top6_"
        num_classes = 95

    log_folder = "../../log/" + data_key + "/"
    
    folder_keyword = "multi_proj_feature_classification/cnn_obj_folder_rf_lda_sum"
    folder_keyword = "cnn_classification"
    folder_name = log_folder + folder_keyword + "/"

    file_keyword = "top45"
    file_keyword = "top15"
    file_keyword = "act3" 
    ###ASL
    file_keyword = "top6"
    file_keyword = "act0_acc_batch_attention_True"
    #file_keyword = "act0_acc_batch_attention_False"
    #file_keyword = "act3_acc_batch_attention_False"
    #file_keyword = "act3_acc_batch_attention_True"

    ###RAR
    #file_keyword = "top30"
    #file_keyword = "top117"


    line_keyword = 'Top Features For Class '
    
    acc_keyword = "[574] cnn_train:"
    #acc_keyword = "Fold eval value"
    train_time_keyword = "[574] cnn_train:"
    test_time_keyword = "[582] cnn_train:"
    results_from_folder(folder_name, file_keyword, acc_keyword, train_time_keyword, test_time_keyword)
    
