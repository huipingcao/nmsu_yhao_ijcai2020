import numpy as np
import os
from os.path import isfile, join, isdir
np.random.seed(1000)

from scipy.io import loadmat

# if len_type == max: extend all the instance attr_len to be equal to the max_len
# if len_type == min: only keep the first min_len values for each attribute
def ucrdata_processing(data_folder, len_type="max"):
    allfiles = os.listdir(data_folder)
    mat_file = ""
    for mat_file in allfiles:
        if mat_file.endswith(".mat") and isfile(join(data_folder, mat_file)):
            break
    if mat_file == "":
        raise Exception("No mat file fould")
    
    data_dict = loadmat(data_folder + mat_file)
    X_train_mat = data_dict['X_train'][0]
    y_train_mat = data_dict['Y_train'][0]
    X_test_mat = data_dict['X_test'][0]
    y_test_mat = data_dict['Y_test'][0]


    y_train = y_train_mat.reshape(-1, 1)
    y_test = y_test_mat.reshape(-1, 1)

    var_list = []
    for i in range(X_train_mat.shape[0]):
        var_count = X_train_mat[i].shape[-1]
        var_list.append(var_count)

    var_list = np.array(var_list)
    max_nb_timesteps = var_list.max()
    min_nb_timesteps = var_list.min()
    median_nb_timesteps = np.median(var_list)
    print('max nb timesteps train : ', max_nb_timesteps)
    print('min nb timesteps train : ', min_nb_timesteps)
    print('median_nb_timesteps nb timesteps train : ', median_nb_timesteps)
    if len_type == "min":
        max_nb_timesteps = min_nb_timesteps


    X_train = np.zeros((X_train_mat.shape[0], X_train_mat[0].shape[0], max_nb_timesteps))

    # pad ending with zeros to get numpy arrays
    for i in range(X_train_mat.shape[0]):
        var_count = X_train_mat[i].shape[-1]
        if var_count == 0:
            continue
        if len_type == "min":
            var_count = min(var_count, min_nb_timesteps)
        X_train[i, :, :var_count] = X_train_mat[i][:, :var_count]

    # ''' Load test set '''

    X_test = np.zeros((X_test_mat.shape[0], X_test_mat[0].shape[0], max_nb_timesteps))

    # pad ending with zeros to get numpy arrays
    for i in range(X_test_mat.shape[0]):
        var_count = X_test_mat[i].shape[-1]
        if var_count == 0:
            continue
        if len_type == "min":
            var_count = min(var_count, min_nb_timesteps)
        X_test[i, :, :var_count] = X_test_mat[i][:, :max_nb_timesteps]

    # ''' Save the datasets '''

    print("Train dataset : ", X_train.shape, y_train.shape)
    print("Test dataset : ", X_test.shape, y_test.shape)
    print("Train dataset metrics : ", X_train.mean(), X_train.std())
    print("Test dataset : ", X_test.mean(), X_test.std())
    print("Nb classes : ", len(np.unique(y_train)))
    print("Max classes : ", max((y_train)))
    min_class = min((y_train))
    y_train = y_train - min_class
    y_test = y_test - min_class
    print("Min train classes : ", min_class)
    print("Min test classes : ", min_class)
    print("Min train classes after minus: ", min((y_train)))
    print("Min test classes after minus: ", min((y_test)))
    return X_train, X_test, y_train, y_test

def file_writingxy(data_x_matrix, data_y_vector, file_name, attr_num=-1, delimiter=' '):
    data_row, data_col = data_x_matrix.shape
    with open(file_name, 'w') as f:
        if attr_num > 0:
            f.write(str(int(attr_num)) + '\n')
        for row in range(0, data_row):
            row_vector = data_x_matrix[row, :]
            row_label = str(int(data_y_vector[row]))
            row_str = row_label
            for index in range(0, data_col):
                row_str = row_str + delimiter + str(row_vector[index])
            f.write(row_str + '\n')


def init_folder(data_folder):
    split_key = '/'
    if data_folder.endswith(split_key):
        data_folder = data_folder[:-1]
    folder_array = data_folder.split(split_key)
    data_folder = ''
    for item in folder_array:
        data_folder = data_folder + item + split_key
        if item == '..':
            continue
        try:
            os.stat(data_folder)
        except:
            os.mkdir(data_folder)

    return data_folder


if __name__ == "__main__":
    data_key = "lstmfcn_dsa"
    #data_key = "lstmfcn_auslan"
    #data_key = "lstmfcn_lp1"
    #data_key = "lstmfcn_htsensor"
    #data_key = "lstmfcn_arabicdigits"
    data_key = "lstmfcn-act"
    len_type = "min"
    len_type = "max"
    data_folder = "../../data/" + data_key +"/"
    X_train, X_test, y_train, y_test = ucrdata_processing(data_folder, len_type)
    x_row, attr_num, attr_len = X_train.shape
    X_train = X_train.reshape(x_row, (attr_num*attr_len))

    x_row, attr_num, attr_len = X_test.shape
    X_test = X_test.reshape(x_row, (attr_num*attr_len))
    out_folder = "../../data/" + data_key + "/train_test_1_fold/"
    out_folder = init_folder(out_folder)
    #file_writingxy(X_train, y_train, out_folder + "train_0.txt", attr_num)
    #file_writingxy(X_test, y_test, out_folder + "test_0.txt", attr_num)