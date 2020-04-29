import numpy as np

class nn_parameters:
    def __init__(self, layer_list, batch_size=100, max_epoch=5, stop_threshold=0.9, activation_fun=0, std_value=0, eval_method='accuracy', save_file='./default_nn.ckpt'):
        self.layer_list = layer_list
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.stop_threshold = stop_threshold
        self.activation_fun = activation_fun
        self.std_value = std_value
        self.eval_method = eval_method
        self.save_file = save_file

    def to_string(self):
        ret_str =  'layer list: \n' + np.array_str(self.layer_list) +'\nbatch size: ' + str(self.batch_size) +'\nmax epoch: ' + str(self.max_epoch) +'\nstop threshold: ' + str(self.stop_threshold)
        if self.activation_fun == 0:
            ret_str = ret_str  + '\nactivation function: RELU'
        elif self.activation_fun == 1:
            ret_str = ret_str  + '\nactivation function: Sigmod'
        else:
            ret_str = ret_str  + '\nactivation function: Tanh'
        ret_str = ret_str +'\ninitial std value: ' + str(self.std_value)
        ret_str = ret_str + '\neval method: ' + self.eval_method
        ret_str = ret_str + '\nsave obj file: ' + self.save_file
        return ret_str


class cnn_parameters:
    # If feature_method == 'none' means do not need to do feature detection
    # conv_kernel_list: [[r1, c1], [r2, c2], [r3, c3]]: means first convolutional kernel is c1 = r1 * c1
    # pool_rate_list: [[r1, c1], [r2, c2], [r3, c3]]: means first pooling kernel is r1 * c1
    # feature_num_list: [a, b], means after c1, there are a numbers of feature maps. b feature maps after c2
    def __init__(self, conv_kernel_list, pool_rate_list, feature_num_list, batch_size=100, max_iter=900, stop_threshold=0.9, activation_fun=0, std_value=0, same_size=False, feature_method='vote', eval_method='accuracy', out_obj_folder='./', out_model_folder='./', group_list = [], input_map=1, full_feature_num=-1, l2_bool=False, keep_prob=0.5):
        self.conv_kernel_list = conv_kernel_list
        self.pool_rate_list = pool_rate_list
        self.feature_num_list = feature_num_list
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.stop_threshold = stop_threshold
        self.activation_fun = activation_fun
        self.std_value = std_value
        self.same_size = same_size
        self.feature_method = feature_method
        self.out_obj_folder = out_obj_folder
        self.out_model_folder = out_model_folder
        self.eval_method = eval_method
        self.input_map = input_map
        self.full_feature_num = full_feature_num
        self.l2_bool = l2_bool
        self.keep_prob_val = keep_prob
    def to_string(self):
        ret_str =  'conv kernel list: \n' + np.array_str(self.conv_kernel_list) +'\npool rate list: \n' + np.array_str(self.pool_rate_list) +'\nfeature map num list: \n' + np.array_str(self.feature_num_list) +'\nbatch size: ' + str(self.batch_size) +'\nmax iteration: ' + str(self.max_iter) +'\nstop threshold: ' + str(self.stop_threshold)
        if self.activation_fun == 0:
            ret_str = ret_str  + '\nactivation function: RELU with count'
        elif self.activation_fun == 1:
            ret_str = ret_str  + '\nactivation function: Sigmod'
        elif self.activation_fun == 2:
            ret_str = ret_str  + '\nactivation function: Tanh'
        else:
            ret_str = ret_str  + '\nactivation function: RELU'
        ret_str = ret_str +'\ninitial std value: ' + str(self.std_value) +'\ncnn same size or not: ' + str(self.same_size) +'\nfeature method: ' + str(self.feature_method)
        ret_str = ret_str + '\neval method: ' + self.eval_method
        ret_str = ret_str + '\nsave obj folder: ' + self.out_obj_folder + '\ntemp obj folder: ' + self.out_model_folder
        #ret_str = ret_str + "\ngroup list: " + str(self.group_list)
        ret_str = ret_str + "\nkeep prob: " + str(self.keep_prob_val)
        return ret_str

def cnn_setting_clone(cnn_setting):
    return cnn_parameters(cnn_setting.conv_kernel_list, cnn_setting.pool_rate_list, cnn_setting.feature_num_list, cnn_setting.batch_size, cnn_setting.max_iter, cnn_setting.stop_threshold, cnn_setting.activation_fun, cnn_setting.std_value, cnn_setting.same_size, cnn_setting.feature_method, cnn_setting.eval_method, cnn_setting.out_obj_folder, cnn_setting.out_model_folder)




#This function is used to conver input string to numpy array, used for conv layers and pooling layers
#The output array column number need to be given, col_num=2 for conv_layers and pooling_layers, col_num=1 for feature_num_list
def string_array_to_numpy_array(input_str_array, delimiter=' ', col_num=2):
    array_len = len(input_str_array)
    return_array = []
    for i in range(0, array_len):
        return_element = []
        element = input_str_array[i].strip()
        element = element.split(delimiter)
        if len(element) != col_num:
            raise Exception("The column number should be " + str(col_num) + " , please check your cnn setting parameter file")
        for item in element:
            element_str = str(item)
            if element_str.startswith('['):
                element_str = element_str.replace('[', '').replace(']','')
                return_element.append(element_str.split(','))
            else:
                return_element.append(int(element_str))
        return_array.append(return_element)
    return_array = np.array(return_array)
    if col_num == 1:
        return_array = return_array.reshape(array_len)
    return return_array

###############################################################
# CNN function



def return_cnn_setting_from_file(cnn_setting_file):
    keyword_list = ['#kernel_list', '#pooling_list', '#feature_num_list', '#batch', '#max_iter', '#stop_threshold', '#activation_func', '#std_value', '#same_size', '#feature_method', '#eval_method']
    
    first_delimiter = ', '
    second_delimiter = ' '

    keyword = ''
    conv_kernel_list = ''
    pool_rate_list = ''
    feature_num_list = ''
    batch_size = -1
    max_iter = -1
    stop_threshold = -1
    activation_func = -1
    std_value = -1
    same_size = ''
    feature_method = ''
    eval_method = ''
    full_feature_num = 400

    lines = open(cnn_setting_file).readlines()
    for line in lines:
        if line.startswith('#'):
            temp = line.strip()
            if temp in keyword_list:
                keyword = line.strip()
            continue
        if keyword == '#kernel_list':
            conv_kernel_list = line.strip()
        elif keyword == '#pooling_list':
            pool_rate_list = line.strip()
        elif keyword == '#feature_num_list':
            feature_num_list = line.strip()
        elif keyword == '#batch':
            batch_size = int(line.strip())
        elif keyword == '#max_iter':
            max_iter = int(line.strip())
        elif keyword == '#stop_threshold':
            stop_threshold = float(line.strip())
        elif keyword == '#activation_func':
            activation_func = int(line.strip())
        elif keyword == '#std_value':
            std_value = float(line.strip())
        elif keyword == '#same_size':
            same_size = line.strip()
        elif keyword == '#feature_method':
            feature_method = line.strip()
        elif keyword == '#eval_method':
            eval_method = line.strip()

    if batch_size<0 or max_iter<0 or stop_threshold<0 or activation_func<0 or std_value<0:
        raise Exception("Wrong data paramters, please check the parameter file " + parameter_file)

    conv_kernel_list = conv_kernel_list.split(first_delimiter)
    pool_rate_list = pool_rate_list.split(first_delimiter)
    feature_num_list = feature_num_list.split(first_delimiter)
    if len(conv_kernel_list) != len(pool_rate_list) or len(pool_rate_list) != len(feature_num_list):
        raise Exception("Conv layers and pooling layers need to have the same number")

    conv_kernel_array = string_array_to_numpy_array(conv_kernel_list, second_delimiter, 2)
    pool_rate_array = string_array_to_numpy_array(pool_rate_list, second_delimiter, 2)
    feature_num_array = string_array_to_numpy_array(feature_num_list, second_delimiter, 1)

    return cnn_parameters(conv_kernel_array, pool_rate_array, feature_num_array, batch_size, max_iter, stop_threshold, activation_func, std_value, same_size, feature_method, eval_method)



def return_cnn_keyword(cnn_setting):
    conv_kernel_list = cnn_setting.conv_kernel_list
    pool_rate_list = cnn_setting.pool_rate_list
    feature_num_list = cnn_setting.feature_num_list
    cnn_keyword = ""
    feature_count = 0
    for item in conv_kernel_list:
        cnn_keyword = cnn_keyword + '_c' + str(item[0]) + '-' + str(item[1]) + '_' + str(feature_num_list[feature_count])
        feature_count = feature_count + 1
    for item in pool_rate_list:
        cnn_keyword = cnn_keyword + '_p' + str(item[0]) + '-' + str(item[1])
    cnn_keyword = cnn_keyword + "_a" + str(cnn_setting.activation_fun)
    return cnn_keyword

def return_cnn_default_setting(conv_kernel_array=np.array([[1, 2], [1, 2]]), pool_rate_array=np.array([[1, 2], [1, 2]]), feature_num_array=np.array([2, 2]), batch_size=100, max_iter=200, stop_threshold=0.9, activation_func=0, std_value=0.02, same_size=False, feature_method='none', eval_method='acc'):
    cnn_setting = cnn_parameters(conv_kernel_array, pool_rate_array, feature_num_array, batch_size, max_iter, stop_threshold, activation_func, std_value, same_size, feature_method, eval_method)
    cnn_keyword = return_cnn_keyword(cnn_setting)
    return cnn_setting, cnn_keyword

# End of CNN function
###############################################################


###############################################################
# NN function

def return_nn_keyword(nn_setting):
    layer_list = nn_setting.layer_list
    nn_keyword = ""
    feature_count = 0
    for item in layer_list:
        nn_keyword = nn_keyword + '_l' + str(item)
        feature_count = feature_count + 1
    
    return nn_keyword

def return_nn_setting_from_file(nn_setting_file):
    keyword_list = ['#layer_list', '#batch_size', '#max_epoch', '#stop_threshold', '#activation_fun', '#std_value', '#eval_method', '#obj_folder']
    
    first_delimiter = ','
    second_delimiter = ' '

    keyword = ''
    layer_list = ''
    batch_size = -1
    max_epoch = -1
    stop_threshold = -1
    activation_fun = -1
    std_value = -1
    eval_method = ''


    lines = open(nn_setting_file).readlines()
    for line in lines:
        if line.startswith('#'):
            temp = line.strip()
            if temp in keyword_list:
                keyword = line.strip()
            continue
        if keyword == '#layer_list':
            layer_list = line.strip()
        elif keyword == '#batch_size':
            batch_size = int(line.strip())
        elif keyword == '#max_epoch':
            max_epoch = int(line.strip())
        elif keyword == '#stop_threshold':
            stop_threshold = float(line.strip())
        elif keyword == '#activation_fun':
            activation_fun = int(line.strip())
        elif keyword == '#std_value':
            std_value = float(line.strip())
        elif keyword == '#eval_method':
            eval_method = line.strip()

    if batch_size<0 or max_epoch<0 or stop_threshold<0 or activation_fun<0 or std_value<0:
        raise Exception("Wrong data paramters, please check the parameter file " + parameter_file)

    layer_list = layer_list.split(first_delimiter)
    layer_list = string_array_to_numpy_array(layer_list, second_delimiter, 1)
    nn_setting = nn_parameters(layer_list, batch_size, max_epoch, stop_threshold, activation_fun, std_value, eval_method)
    nn_keyword = return_nn_keyword(nn_setting)
    nn_setting.save_file = nn_keyword+'.ckpt'
    return nn_setting, nn_keyword


# End of NN function
###############################################################
if __name__ == '__main__':
    #parameter_file = '../../parameters/global_feature_generation.txt'
    #read_global_feature_generation_parameter(parameter_file)

    #parameter_file = '../../parameters/cnn_model_parameter.txt'
    #cnn_p = return_cnn_setting_from_file(parameter_file)
    #print cnn_p.print_to_string()
    #print return_cnn_keyword(cnn_p)

  
    parameter_file = '../../parameters/cnn_model_parameter_varying.txt'
    cnn_p = return_cnn_setting_from_file(parameter_file)
    
