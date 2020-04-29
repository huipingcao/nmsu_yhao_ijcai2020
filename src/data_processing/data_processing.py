import numpy as np
import sys
import os

class data_collection:
    train_x_matrix = None
    train_y_vector = None
    valid_x_matrix = None
    valid_y_vector = None
    test_x_matrix = None
    test_y_vector = None

    train_y_matrix = None
    valid_y_matrix = None
    test_y_matrix = None

    num_classes = 0
    min_class = 0

    class_column = 0
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __init__(self, train_x_matrix, train_y_vector, class_column=0):
        self.train_x_matrix = train_x_matrix
        self.train_y_vector = train_y_vector
        self.class_column = class_column

    def gene_data_stru(self):
        train_x_matrix = self.train_x_matrix
        if train_x_matrix is None:
            raise Exception("Missing training data")
        train_y_vector = self.train_y_vector
        train_shape_len = len(train_x_matrix.shape)
        if train_shape_len == 3:
            train_ins, attr_len, attr_num = train_x_matrix.shape
            input_map = 1
            train_x_matrix = train_x_matrix.reshape(train_ins, attr_len, attr_num, input_map)
        elif train_shape_len == 4:
            train_ins, attr_len, attr_num, input_map = train_x_matrix.shape
        else:
            raise Exception("Input x matrix invalid shape!!!")
        min_class = min(train_y_vector)
        max_class = max(train_y_vector)
        num_classes = max_class - min_class + 1
        return return_data_stru(num_classes, min_class, attr_num, attr_len, self.class_column, train_ins)

    def data_check(self, num_classes, min_class):
        if self.train_y_vector is not None:
            self.train_y_matrix = y_vector_to_matrix(self.train_y_vector, num_classes, min_class)
        if self.test_y_vector is not None:
            self.test_y_matrix = y_vector_to_matrix(self.test_y_vector, num_classes, min_class)
        if self.valid_y_vector is not None:
            self.valid_y_matrix = y_vector_to_matrix(self.valid_y_vector, num_classes, min_class)


########################################################################################
## data structure part
# starting from min_class, max_class = min_class + num_classes - 1
# It means that all numbers between min_class to max_class should be used as a label
class data_structure:
    def __init__(self, num_classes, min_class, attr_num, attr_len, class_c=0, train_ins=-1):
        self.num_classes = num_classes
        self.attr_num = attr_num
        self.attr_len = attr_len
        self.class_column = class_c
        self.train_ins = train_ins
        self.min_class = min_class
    def print_to_string(self):
        ret_str =  'num of classes: ' + str(self.num_classes) +'\nattribute number: ' + str(self.attr_num) +'\nattribute length: ' + str(self.attr_len)  +'\nclass column: ' + str(self.class_column) +'\ntrain_ins: ' + str(self.train_ins) 
        return ret_str


def return_data_stru(num_classes, min_class, attr_num, attr_len, class_column, train_ins=-1):
    return data_structure(num_classes, min_class, attr_num, attr_len, class_column, train_ins)


def data_stru_gene(train_y_vector, class_colum=0):
    min_class = min(train_y_vector)
    max_class = max(train_y_vector)
    num_clasess = max_class - min_class + 1




def copy_data_stru(in_data_stru):
    return data_structure(in_data_stru.num_classes, in_data_stru.start_class, in_data_stru.attr_num, in_data_stru.attr_len)


## end of data structure part
########################################################################################



def train_test_transpose(data_matrix, attr_num, attr_len, trans=True):
    data_row, data_col = data_matrix.shape
    data_matrix = data_matrix.reshape(data_row, attr_num, attr_len, 1)
    #data_matrix = data_matrix.reshape(data_row, attr_num, 1, attr_len)
    if trans == True:
        data_matrix = np.transpose(data_matrix, (0, 2, 3, 1))
    else:
        data_matrix = np.transpose(data_matrix, (0, 2, 1, 3))
    #data_matrix = data_matrix.reshape(data_row, data_col)
    return data_matrix

def y_vector_to_matrix(y_vector, num_classes, start_class=0):
    vector_len = len(y_vector)
 #   print y_vector
 #   print vector_len
 #   print num_classes
 #   print "========"
    y_matrix = np.zeros((vector_len, num_classes))
    count = 0
    for item in y_vector:
        y_matrix[count, int(item)-start_class] = int(1)
        count = count + 1
    return y_matrix

def class_label_vector_checking(y_vector):
    min_class = min(y_vector)
    max_class = max(y_vector)
    class_index_dict = {}
    min_length = -1
    max_length = -1
    for c in range(min_class, max_class+1):
        c_index = np.where(y_vector==c)[0]
        class_index_dict[c] = c_index
        if min_length == -1:
            min_length = len(c_index)
        elif len(c_index) < min_length:
            min_length = len(c_index)
        if max_length == -1:
            max_length = len(c_index)
        elif len(c_index) > max_length:
            max_length = len(c_index)

    return class_index_dict, min_length, max_length


def feature_data_generation_4d(data_matrix, feature_index_list):
    row_n, attr_len, num_map, attr_num = data_matrix.shape
    
    ret_matrix = []
    new_row_col = 0
    
    new_attr = len(feature_index_list)

    new_row_col = new_attr * attr_len
    for i in range(0, row_n):
        ori_matrix = data_matrix[i].reshape(attr_len, attr_num)
        matrix = ori_matrix[:, feature_index_list]
        ret_matrix.append(matrix.reshape(new_row_col))
    
    data_matrix = np.array(ret_matrix).reshape(row_n, new_row_col)

    return np.array(ret_matrix).reshape(row_n, new_row_col), new_attr

def feature_data_generation(data_matrix, attr_len, attr_num, feature_index_list):
    row_n, col_n = data_matrix.shape
    ret_matrix = []
    new_row_col = 0
    
    new_attr = len(feature_index_list)

    new_row_col = new_attr * attr_len
    for i in range(0, row_n):
        ori_matrix = data_matrix[i].reshape(attr_len, attr_num)
        matrix = ori_matrix[:, feature_index_list]
        ret_matrix.append(matrix.reshape(new_row_col))
    
    data_matrix = np.array(ret_matrix).reshape(row_n, new_row_col)

    return np.array(ret_matrix).reshape(row_n, new_row_col), new_attr


def feature_data_generation_v1(data_matrix, attr_num, feature_index_list, group_list=[]):
    row_n, col_n = data_matrix.shape
    attr_len = col_n/attr_num
    ret_matrix = []
    new_row_col = 0
    
    new_attr = len(feature_index_list)

    if len(group_list) > 0:
        for group in group_list:
            new_attr = new_attr + len(group)

    new_row_col = new_attr * attr_len

    for i in range(0, row_n):
        ori_matrix = data_matrix[i].reshape(attr_num, attr_len)
        if len(group_list) > 0:
            group_count = 0
            for group in group_list:
                if group_count == 0:
                    matrix = ori_matrix[group, :]
                else:
                    matrix = np.append(matrix, ori_matrix[group, :])
                group_count = group_count + 1
            matrix = np.append(matrix, ori_matrix[feature_index_list, :])
        else:
            matrix = ori_matrix[feature_index_list, :]
        ret_matrix.append(matrix.reshape(new_row_col))
    
    data_matrix = np.array(ret_matrix).reshape(row_n, new_row_col)

    return np.array(ret_matrix).reshape(row_n, new_row_col), new_attr, attr_len


def z_normlization(time_series):
    mean = np.mean(time_series)
    dev = np.std(time_series)
    return (time_series - mean)/dev


if __name__ == '__main__':
    series1 = [2.02, 2.33, 2.99, 6.85, 9.20, 8.80, 7.50, 6.00, 5.85, 3.85, 4.85, 3.85, 2.22, 1.45, 1.34]
    series2 = [-0.12, -0.16, -0.13,  0.28,  0.37,  0.39,  0.18,  0.09,  0.15, -0.06,  0.06, -0.07, -0.13, -0.18, -0.26]
    norm_1 = z_normlization(series1)
    norm_2 = z_normlization(series2)
    x = range(0, len(series1))
    
    #data_str = 'uci'
    #uci_data_stru = return_data_stru(data_str)
    ##uci_data_stru.num_classes = 3
    ##uci_data_stru.start_class = 11
    #uci_data_stru.print_to_string()
#
    row_num = 1
    data_matrix = np.random.rand(row_num, 24)
    max_gap = 0
    attr_num=6
    feature_index_list = np.array([row_num, 3])
    print (data_matrix.reshape(row_num, attr_num, 4))

    model = Generation_model(attr_num, 3, [], 2)

    matrix, attr_num, attr_len = feature_data_generation(data_matrix, attr_num, model.selected_list, model.groups)
    print ("===")
    print (model.selected_list)
    print (model.groups)
    print (matrix.reshape(row_num, attr_num, attr_len))


###########################################################################
## Duplication part, used for multiple time series data. Do duplication on attributes dimension in order to run CNN
## duplicate rows
#def row_duplication(data_matrix, max_gap):
#    ret_matrix = data_matrix.copy()
#    row_n, col_n = ret_matrix.shape
#    for i in range(2, max_gap+1):
#        ret_matrix = np.append(ret_matrix, data_matrix[::i], axis=0)
#        ret_matrix = np.append(ret_matrix, data_matrix[1::i], axis=0)
#    return ret_matrix
#
#
## duplicate cols
#def col_duplication(data_matrix, max_gap):
#    ret_matrix = data_matrix.copy()
#    row_n, col_n = ret_matrix.shape
#    for i in range(2, max_gap+1):
#        ret_matrix = np.append(ret_matrix, data_matrix[:, ::i], axis=1)
#        ret_matrix = np.append(ret_matrix, data_matrix[:, 1::i], axis=1)
#    return ret_matrix
#
#
## Use to update data matrix in order to generate features based on time dimension
## data_matrix: A * T: A is number of attributes, T is the length of time dimension
## max_gap: In order to run cnn feature detection, we would like to do duplication on attribute dimension. No duplication if map_gap ==0
## data_stru: information for data_matrix
## Logic: 1, do matrix transpose to get data_matrix T * A
## 2, in order to get rid of the effect from attribute order, we do duplication on attribute dimension
## result from step 2 is T * (A + A/2 + A/3 + ... until max_gap) 
## return updated data_matrix and updated data_stru
#def time_as_feature_transpose(data_matrix, max_gap):
#    data_matrix = np.transpose(data_matrix) # T * A
#    if max_gap == 0:
#        return data_matrix
#
#    data_matrix = col_duplication(data_matrix, max_gap) # data_matrix with duplication on attribute dimension (column dimension)
#    return data_matrix
#
#
## We need d3_time_as_feature_transpose because the original data matrix is N * (A * T), N is number of instances, and column lenght is (A * T) which is the number of attributes times attribute length
#def d3_time_as_feature_transpose(d3_data_matrix, max_gap, data_stru):
#    attr_num = data_stru.attr_num
#    attr_len = data_stru.attr_len # which is the length of time dimension now
#
#    row_n, col_n = d3_data_matrix.shape
#    ret_data_matrix = list()
#    data_row_matrix = d3_data_matrix[0].reshape(attr_num, attr_len)
#    data_row_matrix = time_as_feature_transpose(data_row_matrix, max_gap)
#    new_row, new_col = data_row_matrix.shape
#    new_row_col_all = new_row * new_col
#    ret_data_matrix.append(data_row_matrix.reshape(new_row_col_all))
#    for i in range(1, row_n):
#        data_row_matrix = d3_data_matrix[i].reshape(attr_num, attr_len)
#        data_row_matrix = time_as_feature_transpose(data_row_matrix, max_gap)
#        ret_data_matrix.append(data_row_matrix.reshape(new_row_col_all))
#
#    ret_data_matrix = np.array(ret_data_matrix).reshape(row_n, new_row_col_all)
#    ret_data_stru = data_structure(data_stru.num_classes, data_stru.start_class, new_row, new_col, data_stru.class_column)
#
#    return ret_data_matrix, ret_data_stru
#
##End of Duplication part
###########################################################################
#
#
###########################################################################
##cross validataion part
#
##Given data matrix (x_matrix) and correspsonding class label vector (y_vector), do cross validation
##Need num_classes to make sure the validataion is balanced for all classes
##ratio: the ratio of testing data
#def cross_validation(x_matrix, y_vector, num_classes, ratio=0.1):
#    instance_count = len(y_vector)
#    one_class_count = instance_count/num_classes
#    start = 0;
#    end = start + one_class_count
#    train_x_matrix, test_x_matrix, train_y_vector, test_y_vector = train_test_split(x_matrix[start:end, :], y_vector[start:end], test_size=ratio, random_state=0)
#    start = end
#    end = end + one_class_count
#    while(end<=instance_count):
#        sub_train_x, sub_test_x, sub_train_y, sub_test_y = train_test_split(x_matrix[start:end, :], y_vector[start:end], test_size=ratio, random_state=0)
#        train_x_matrix = np.concatenate((train_x_matrix, sub_train_x), axis = 0)
#        test_x_matrix = np.concatenate((test_x_matrix, sub_test_x), axis=0)
#        train_y_vector.extend(sub_train_y)
#        test_y_vector.extend(sub_test_y)
#        start = end
#        end = end + one_class_count
#    return train_x_matrix, train_y_vector, test_x_matrix, test_y_vector
#
##End of cross validation part
###########################################################################
#
#
###########################################################################
##Using feature to generate partial data
#
##For multiple time series data matrix
##data_matrix: N * M: N is the number of instances, M is a vector to represnet the attr * time matrix
##Need attr_num to reshape the 1*M vector to attr * time matrix
##attr_index_list: numpy array for the key attribute indexes
##time_index_list: numpy array for the key time indexes
#def old_feature_data_generation(data_matrix, attr_num, attr_index_list = None, method='attribute', time_index_list = None):
#    row_n, col_n = data_matrix.shape
#    time_len = col_n/attr_num
#    ret_matrix = []
#    new_row_col = 0
#    if method == 'attribute':
#        new_row = len(attr_index_list)
#        new_row_col = new_row * time_len
#        for i in range(0, row_n):
#            matrix = data_matrix[i].reshape(attr_num, time_len)
#            matrix = matrix[attr_index_list, :]
#            ret_matrix.append(matrix.reshape(new_row_col))
#        attr_num = new_row
#    elif method == 'time':
#        new_col = len(time_index_list)
#        new_row_col = attr_num * new_col
#        for i in range(0, row_n):
#            matrix = data_matrix[i].reshape(attr_num, time_len)
#            matrix = matrix[:, time_index_list]
#            ret_matrix.append(matrix.reshape(new_row_col))
#        time_len = new_col
#    return np.array(ret_matrix).reshape(row_n, new_row_col), attr_num, time_len
#
## input data_matrix: 2d matrix with r * (a * l). 
## r number of instances and each instance has a attributes and each attribute has length l
## atr_num: attribute number
## fature_index_list: a list contains the index of picked attributes
## Rturn: return a data matrix only contains the attributes from feature_index_list
#def feature_data_generation(data_matrix, attr_num, feature_index_list, feature_col_update=True):
#    row_n, col_n = data_matrix.shape
#    attr_len = col_n/attr_num
#    ret_matrix = []
#    new_row_col = 0
#    
#    if feature_col_update == True:
#        new_row = len(feature_index_list)
#        new_row_col = new_row * attr_len
#    else:
#        new_row = attr_num
#        new_row_col = col_n
#
#    for i in range(0, row_n):
#        matrix = data_matrix[i].reshape(attr_num, attr_len)
#        if feature_col_update == True:
#            matrix = matrix[feature_index_list, :]
#            ret_matrix.append(matrix.reshape(new_row_col))
#        else:
#            a = range(0, attr_num)
#            remove_index = [x for x in a if x not in feature_index_list]
#            matrix[remove_index, :] = 0
#            ret_matrix.append(matrix.reshape(new_row_col))
#
#    attr_num = new_row
#    return np.array(ret_matrix).reshape(row_n, new_row_col), attr_num, attr_len
#
#
#
#
#
#
#
#def class_label_vector_checking(y_vector):
#    min_class = min(y_vector)
#    max_class = max(y_vector)
#    class_index_dict = {}
#    min_length = -1
#    max_length = -1
#    for c in range(min_class, max_class+1):
#        c_index = np.where(y_vector==c)[0]
#        class_index_dict[c] = c_index
#        if min_length == -1:
#            min_length = len(c_index)
#        elif len(c_index) < min_length:
#            min_length = len(c_index)
#        if max_length == -1:
#            max_length = len(c_index)
#        elif len(c_index) > max_length:
#            max_length = len(c_index)
#
#    return class_index_dict, min_length, max_length




