import logging
import os
from os.path import isfile, join
import numpy as np
from data_io import file_reading
from data_io import x_y_spliting
#import matplotlib.pyplot as plt

def data_plot(data_file, class_column=0, delimiter=' '):
    x_matrix, attr_num = file_reading(data_file, delimiter, True)
    x_matrix, y_vector = x_y_spliting(x_matrix, class_column)
    y_min = min(y_vector)
    y_max = max(y_vector)
    x_row, x_col = x_matrix.shape
    attr_len = x_col/attr_num
    x_matrix = x_matrix.reshape(x_row, attr_num, attr_len)
    for label in range(y_min, y_max):
        out_pdf = "asl_class_" + str(label) + ".pdf"
        fig = plt.figure()

        label_index = np.where(y_vector==label)[0]
        label_row = x_matrix[label_index[0], :, :]
        
        for attr in range(0, attr_num):
            plot_series = label_row[attr, :]
            plot_len = len(plot_series)
            stop_i = plot_len
            for i in range(0, plot_len):
                re_i = plot_len - i - 1
                if plot_series[re_i] == 0:
                    stop_i = stop_i - 1
                else:
                    break
            
            plt.plot(plot_series[0:stop_i])
        fig.savefig(out_pdf, dpi=fig.dpi)


def data_checking(data_file, class_column=0, delimiter=' '):
    ret_str = ""
    x_matrix, attr_num = file_reading(data_file, delimiter, True)
    x_matrix, y_vector = x_y_spliting(x_matrix, class_column)
    ret_str = 'x_matrix shape: ' + str(x_matrix.shape)
    y_min = min(y_vector)
    y_max = max(y_vector)


    ret_str = ret_str + "\nclass labels from " + str(y_min) + " to " + str(y_max)
    #for i in range(y_min, y_max+1):
    #    ret_str = ret_str + '\nclass '+ str(i) + ': '+str(y_vector.count(i))
    unique, counts = np.unique(y_vector, return_counts=True)
    ret_str = ret_str +'\n'+ str(dict(zip(unique, counts)))
    return ret_str

def arc_reduce_null(fname, null_class=1, null_max=1000, class_column=0, delimiter=' ', header = True):
    num = 0
    data_matrix = []
    null_count = 0
    with open(fname) as f:
        data_row = []
        for line in f:
            if header == True:
                attr_num = int(line.strip())
                header = False
                continue
            data_row = line.split(delimiter)
            if int(data_row[class_column]) == null_class:
                null_count = null_count + 1
                if null_count <  null_max:
                    data_matrix.append(data_row)
            else:
                data_matrix.append(data_row)

    row_num = len(data_matrix)
    col_num = len(data_matrix[0])
    data_matrix = np.array(data_matrix, dtype=float).reshape(row_num, col_num)
    data_matrix.astype(float)
    y_vector = data_matrix[:, class_column].astype(int)
    return np.delete(data_matrix, class_column, 1), y_vector





if __name__ == '__main__':
    #data_file = '../../data/gesture_data/processed_data/data.txt_trainTest10/train_0.txt'
    #data_file = '../../data/arc_activity_recognition/s1_ijcal/train.txt'
    #class_column = 0
    #delimiter = ' '
    #ret_str = data_checking(data_file, class_column, delimiter)
    #print ret_str
    #data_file = '../../data/arc_activity_recognition/s1_ijcal/test.txt'
    #class_column = 0
    #delimiter = ' '
    #ret_str = data_checking(data_file, class_column, delimiter)
    #print ret_str
    data_file = '../../data/evn/ds/DS_all_ready_to_model.csv_trainTest2_weekly_3attr/test_0.txt'
    #data_file = '../../data/human/subject10_ideal.log'
    #class_column = 119
    #delimiter = '\t'
    ##null_class=1
    ##null_max=1000
    ##x_matrix, y_vector = readFile(data_file, null_class, null_max, class_column);
    ##print x_matrix.shape
    ##print y_vector.shape
#
    #data_file = '../../data/human/processed/ready/data.txt'#_trainTest10/train_0.txt'
    #class_column = 0
    #delimiter = ' '
    #ret_str = data_checking(data_file, class_column, delimiter)
    #print ret_str

    data_file = '../../data/dsa/train_test_10_fold/test_0.txt'
    #data_file = '../../data/dsa/output.txt'
    #data_file = '../../data/rar/train_test_10_fold_class_based/train_0.txt_class_0.txt'
    #data_file = "../../data/arabic/train_test_1_fold/train_0.txt"
    #data_file = "../../data/arabic/train_test_1_fold/test_0.txt"
    #data_file = "../../data/asl/train_test_3_fold/train_0.txt"
    #data_file = '../../data/rar/train_test_10_fold/test_0.txt'
    #data_file = '../../data/arc/train_test_10_fold/test_0.txt'
    #data_file = '../../data/fixed_arc/train_test_1_fold/test_0.txt'
    data_key = "phs"
    data_key = "eeg"
    #data_key = "fad"
    data_file = "../../data/" + data_key +"/train.txt"
    class_column = 0
    delimiter = ' '
    #data_plot(data_file, class_column, delimiter)
    ret_str = data_checking(data_file, class_column, delimiter)
    print(ret_str)
    
