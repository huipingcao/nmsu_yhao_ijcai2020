import logging
import numpy as np
import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'fileio/'))
from data_io import list_files
from data_io import write_to_excel
import os


def results_from_file(file_name, line_keyword, bias=0):
    global confirm_keyword
    if confirm_keyword == "":
        confirm_need = False
    else:
        confirm_need = True
    keyword_len = len(line_keyword)
    confirm_len = len(confirm_keyword)
    count = 0
    #value_matrix = []
    value_vector = []
    confirm_value = 0
    previous = -1
    start = False
    end = False
    train_keyword = line_keyword.replace('f1 for class', 'fold training time')
    test_keyword = line_keyword.replace('f1 for class', 'fold testing time')
    acc_keyword = line_keyword.replace('f1 for class', 'fold accuracy:')

    accuracy = -1
    train_time = 0
    test_time = 0
    with open(file_name) as f:
        value_vector = []
        for line in f:
            if confirm_need is True:
                if confirm_keyword in line:
                    confirm_value = float(line[(line.index(confirm_keyword) + confirm_len):])
            
            if acc_keyword in line:
                line_array = line.split(':')
                accuracy = float(line_array[-1])
            elif train_keyword in line:
                line_array = line.split(':')
                time_vector = line_array[-1].replace('[', '').replace(']', '')
                time_vector = time_vector.split(',')
                time_vector = [float(a) for a in time_vector]
                train_time = sum(time_vector)
            elif test_keyword in line:
                line_array = line.split(':')
                time_vector = line_array[-1].replace('[', '').replace(']', '')
                time_vector = time_vector.split(',')
                time_vector = [float(a) for a in time_vector]
                test_time = sum(time_vector)
            if line_keyword not in line:
                continue
            
            line = line.strip()
            start = line.index(line_keyword) + keyword_len
            line = line[start:]
            line_array = line.split(':')
            line_value = line_array[-1]
            nextLine = line
            while "INFO" not in nextLine:
                line_value = line_value + nextLine
                try:
                    nextLine = next(f)
                except:
                    break
            line_value = line_value.replace('[', '').replace(']','')
            temp_line_array = line_value.split(' ')

            for item in temp_line_array:
                try:
                    item = float(item)
                    value_vector.append(item)
                except:
                    pass
            #value_vector = np.array(value_vector)
            #value_matrix.append(value_vector)
    #value_matrix = np.array(value_matrix)
    return value_vector, accuracy, train_time, test_time

def results_from_folder(folder_name, file_keyword, num_classes, line_keyword, bias=0):
    file_list = list_files(folder_name)
    value_matrix = []
    file_count = -1
    file_count_vector = []
    file_count_vector.append(file_count)
    accuracy_vector = []
    train_time_vector = []
    test_time_vector = []
    for file_name in file_list:
        if file_name.startswith('.'):
            continue
        if file_keyword not in file_name:
            continue
        print(file_name)
        file_count = file_count + 1
        file_count_vector.append(file_name.split('_')[2])
        value_vector, accuracy, train_time, test_time = results_from_file(folder_name+file_name, line_keyword, bias)
        print(np.array(value_vector).shape)
        for add in range(len(value_vector), num_classes):
            value_vector.append(-1)
        value_matrix.append(value_vector)
        accuracy_vector.append(accuracy)
        train_time_vector.append(train_time)
        test_time_vector.append(test_time)
    value_matrix = np.array(value_matrix)
    #file_count_vector = np.array(file_count_vector).astype(int)
    acc_time_matrix = []
    acc_time_matrix.append(train_time_vector)
    acc_time_matrix.append(test_time_vector)
    acc_time_matrix.append(accuracy_vector)
    return value_matrix, file_count, np.array(acc_time_matrix)



if __name__ == '__main__':
    data_key = "dsa"
    #data_key = "rar"
    #data_key = "arc"
    #data_key = "ara"
    #data_key = "asl"
    #data_key = "fixed_arc"
    method = "cnn"
    #method = "knn"
    #method = "libsvm"
    #method = "rf"

    if data_key == "dsa":
        top_k = "_top15_"
        top_k = "_top45_"
        num_classes = 19
    elif data_key == "rar":
        top_k = "_top30_"
        top_k = "_top117_"
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
    
    folder_name = log_folder +  "/cnn_classification/"
    
    file_keyword = ".log_2019_"
    #print file_keyword

    confirm_keyword = ""
    line_keyword = 'final best f1:'
    line_keyword = method + ' f1 for class '
    bias = 3
    class_start = 0
    class_end = num_classes
    
    #value_matrix, index_vector, file_count_vector = results_from_folder_class_based(folder_name, file_keyword,  class_start, class_end, line_keyword, bias)
    value_matrix, file_count, acc_time_matrix = results_from_folder(folder_name, file_keyword, num_classes, line_keyword, bias)
    print("value matrix shape init")
    print(value_matrix.shape)
    value_row, value_col = value_matrix.shape
    file_count_vector = np.array(range(-1, value_row))
    print(file_count_vector)
    file_count_vector = file_count_vector.reshape(len(file_count_vector), 1)
    index_vector = np.array([range(0, value_col)])
    print(index_vector.shape)
    print(file_count_vector.shape)
    write_matrix = np.append(index_vector, value_matrix, axis=0)
    print("value matrix shape sec")
    print(value_matrix.shape)
    write_matrix = np.append(file_count_vector, write_matrix, axis=1)
    print("value matrix shape final")
    print(value_matrix.shape)
    
    to_excel_file = "../../test.xlsx"
    if os.path.exists(to_excel_file):
        os.remove(to_excel_file)
    write_to_excel(write_matrix, to_excel_file)
    acc_time_matrix = acc_time_matrix.transpose()
    print(acc_time_matrix.shape)
    print(file_count_vector.shape)
    acc_time_matrix = np.append(file_count_vector[1:, :], acc_time_matrix, axis=1)

    write_to_excel(acc_time_matrix, to_excel_file, 17, 4)
