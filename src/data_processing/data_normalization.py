import numpy as np
import sys

from data_io import train_test_file_reading_with_attrnum
from data_io import list_files
from data_io import init_folder
from data_io import file_writing
from data_io import file_reading
from data_io import x_y_spliting
from log_io import setup_logger
from data_processing import z_normlization
##
# data_matrix: data_row * attr_num * attr_len
def run_z_normalization(data_matrix):
    data_row, attr_num, attr_len = data_matrix.shape
    for row in range(0, data_row):
        for attr in range(0, attr_num):
            data_matrix[row, attr, :] = z_normlization(data_matrix[row, attr, :])
    return data_matrix


def run_z_norm_main(data_folder, file_keyword="train_", logger=None, class_column=0, delimiter=' ', header=True):
    if logger is None:
        logger = setup_logger('')

    if data_folder.endswith('/'):
        out_folder = data_folder[:-1] + "_z_norm/"
    else:
        out_folder = data_folder + "_z_norm/"
    out_folder = init_folder(out_folder)
    file_list = list_files(data_folder)
    file_count = 0
    for train_file in file_list:
        if file_keyword not in train_file:
            continue
        logger.info(train_file)
        test_file = train_file.replace('train', 'test')
        train_x_matrix, train_y_vector, test_x_matrix, test_y_vector, attr_num = train_test_file_reading_with_attrnum(
            data_folder + train_file, data_folder + test_file, class_column, delimiter, header)
        
        #train_x_matrix = train_x_matrix[0:20, :]
        #test_x_matrix = test_x_matrix[0:20, :]
        #train_y_vector = train_y_vector[0:20]
        #test_y_vector = test_y_vector[0:20]

        train_row, train_col = train_x_matrix.shape
        test_row, test_col = test_x_matrix.shape
        attr_len = train_col/attr_num
        train_x_matrix = train_x_matrix.reshape(train_row, attr_num, attr_len)
        test_x_matrix = test_x_matrix.reshape(test_row, attr_num, attr_len)

        norm_train_matrix = run_z_normalization(train_x_matrix)
        norm_test_matrix = run_z_normalization(test_x_matrix)
        if file_count == 0:
            logger.info("Before norm")
            logger.info('train matrix shape: ' + str(train_x_matrix.shape))
            logger.info('test matrix shape: ' + str(test_x_matrix.shape))
            logger.info("After norm")
            logger.info('train matrix shape: ' + str(norm_train_matrix.shape))
            logger.info('test matrix shape: ' + str(norm_test_matrix.shape))
        norm_train_matrix = norm_train_matrix.reshape(train_row, train_col)
        norm_test_matrix = norm_test_matrix.reshape(test_row, test_col)
        train_y_vector = train_y_vector.reshape(len(train_y_vector), 1)
        test_y_vector = test_y_vector.reshape(len(test_y_vector), 1)
        norm_train_matrix = np.hstack((train_y_vector, norm_train_matrix))
        norm_test_matrix = np.hstack((test_y_vector, norm_test_matrix))
        if file_count == 0:
            logger.info("before write to file")
            logger.info('train matrix shape: ' + str(norm_train_matrix.shape))
            logger.info('test matrix shape: ' + str(norm_test_matrix.shape))
        file_writing(norm_train_matrix, out_folder + train_file, attr_num)
        file_writing(norm_test_matrix, out_folder + test_file, attr_num)
        if norm_checking(out_folder + train_file) is False or norm_checking(out_folder + test_file) is False:
            logger.info("ERROR!!!")
            raise Exception("ERROR!!!")
            return False
        file_count = file_count + 1

def norm_checking(data_file):
    data_matrix, attr_num = file_reading(data_file)
    data_x_matrix, data_y_vector = x_y_spliting(data_matrix, 0)
    data_row, data_col = data_x_matrix.shape
    attr_len = data_col/attr_num
    data_x_matrix = data_x_matrix.reshape(data_row, attr_num, attr_len)
    for row in range(0, data_row):
        for attr in range(0, attr_num):
            series = data_x_matrix[row, attr, :]
            mean = np.mean(series)
            std = np.std(series)
            if mean >0.0001 or mean < -0.0001:
                return False
            if std > 1.00001 or std < 0.99999:
                return False
    return True

if __name__ == '__main__':
    argv_array = sys.argv
    run_stdout = sys.stdout
    file_keyword = 'train_'
    len_argv_array = len(argv_array)
    if len_argv_array > 1:
        try:
            val = int(argv_array[1])
            file_keyword = file_keyword + argv_array[1]
        except ValueError:
            print("That's not an int!")

    #data_matrix = np.random.rand(3, 2, 5)
    #print data_matrix
    #print run_z_normalization(data_matrix).shape
    data_key = 'dsa'
    data_key = 'rar'
    data_key = 'arc'
    data_key = 'arabic'
    data_folder = "../../data/" + data_key + "/train_test_1_fold/"
    log_folder = "../../log/" + data_key
    log_folder = init_folder(log_folder)
    log_file = log_folder + data_key + "_z_norm.log"
    logger = setup_logger(log_file)
    run_z_norm_main(data_folder, file_keyword, logger)
