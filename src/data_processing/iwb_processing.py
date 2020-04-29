import numpy as np
import sys
import os

from plot import plot_2dmatrix

sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'fileio/'))
from data_io import file_writing
from data_io import init_folder
from data_io import list_files
from data_io import file_writingxy

def read_iwb_data(ori_iwb_file, attr_num=22):
    num = 0
    data_x_matrix = []
    data_y_vector = []
    with open(ori_iwb_file) as f:
        for line in f:
            if '@' in line or '%' in line:
                continue
            line = line.strip()
            if len(line) == 0:
                continue
            num = num + 1
            line = line.replace('"', "'")
            line_array = line.split("',")
            attr_line = line_array[0].strip()
            label_str = line_array[1].strip()
            attr_line = attr_line.replace("?", "0")
            attr_line = attr_line.replace("'", "")
            attr_line = attr_line.replace(" ", "")
            attr_array = attr_line.split("\\n")
            attr_matrix = []
            for item in attr_array:
                item = item.strip()
                if len(item) == 0:
                    continue
                item_array = item.split(',')
                #print(item_array)
                #print(len(item_array))
                if len(item_array) != attr_num:
                    raise Exception("Lenth is not correct. Check!!!")
                attr_matrix.append(item_array)
            attr_matrix = np.array(attr_matrix)
            attr_matrix = attr_matrix.T
            data_x_matrix.append(attr_matrix)
            data_y_vector.append(label_str)
    data_x_matrix = np.array(data_x_matrix).astype(float)
    data_y_vector = np.array(data_y_vector)
    return data_x_matrix, data_y_vector


def str_v_to_num_vector(str_y_vector, label_list):
    num_y_vector = np.zeros(len(str_y_vector))
    label_count = 0
    for label in label_list:
        if label_count > 0:
            label_index = np.where(str_y_vector==label)[0]
            num_y_vector[label_index] = label_count
        label_count = label_count + 1
    return num_y_vector

def iwb_processing_main(data_folder):
    attr_num = 22
    output_folder = data_folder + "raw/"
    init_folder(output_folder)
    file_list = list_files(data_folder)
    train_file = ""
    test_file = ""
    for file_name in file_list:
        if "TRAIN" in file_name:
            train_file = file_name
        if "TEST" in file_name:
            test_file = file_name
    if train_file == "" or test_file == "":
        raise Exception("file missing")

    train_x_matrix, train_y_str_vector = read_iwb_data(data_folder+train_file, attr_num)
    test_x_matrix, test_y_str_vector = read_iwb_data(data_folder+test_file, attr_num)
    train_row, train_attr, attr_len = train_x_matrix.shape
    test_row, attr_num, attr_len = test_x_matrix.shape
    print(train_x_matrix.shape)
    print(test_x_matrix.shape)

    # train_zeros = []
    # test_zeros = []
    # for i in range(attr_num):
    #     train_data = train_x_matrix[:, i, :]
    #     print(np.amax(train_data))
    #     if np.amax(train_data) == 0:
    #         train_zeros.append(i)
    #     test_data = test_x_matrix[:, i, :]
    #     if np.amax(test_data) == 0:
    #         test_zeros.append(i)
    # print(train_zeros)
    # print(test_zeros)
    plot_2dmatrix(train_x_matrix[0, 0:2, :].T)
    train_x_matrix = train_x_matrix.reshape(train_row, (attr_num * attr_len))
    test_x_matrix = test_x_matrix.reshape(test_row, (attr_num * attr_len))
    label_list = np.unique(train_y_str_vector)
    train_y_vector = str_v_to_num_vector(train_y_str_vector, label_list)
    test_y_vector = str_v_to_num_vector(test_y_str_vector, label_list)
    
    
    train_out_file = "train_0.txt"
    test_out_file = "test_0.txt"
    #file_writingxy(train_x_matrix, train_y_vector, output_folder + train_out_file, attr_num, ' ')
    #file_writingxy(test_x_matrix, test_y_vector, output_folder + test_out_file, attr_num, ' ')



if __name__ == '__main__':
    argv_array = sys.argv
    run_stdout = sys.stdout
    file_keyword = 'train_'
    len_argv_array = len(argv_array)

    data_key = "fad"
    data_key = "jav"
    data_key = "iwb"
    iwb_folder = "../../data/iwb/"
    iwb_folder = "../../data/asd/"
    iwb_folder = "../../data/phs/"
    iwb_folder = "../../data/" + data_key + "/"
    iwb_processing_main(iwb_folder)
