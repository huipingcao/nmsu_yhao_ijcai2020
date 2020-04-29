from .data_io import init_folder
from .data_io import check_folder_exists
from .data_io import check_file_exists



# The function is used to read the input method key file
# The key file contains the data file we need to process
def read_genetic_cnn_parameter(parameter_file, function_keyword="genetic_cnn"):
    lines = open(parameter_file).readlines()
    split_key = '/'

    keyword_list = ['#data_sub_folder', '#data_keyword', '#data_folder', '#attr_num', '#attr_len', '#num_classes', '#start_class', '#class_column', '#method', '#log_folder', '#obj_folder', '#out_model_folder', '#cnn_setting_file']

    keyword = ''
    data_keyword = ''
    data_folder = ''
    attr_num = -1
    attr_len = -1
    num_classes = -1
    start_class = -1
    class_column = -1
    log_folder = ''
    obj_folder = ''
    out_model_folder = ''
    cnn_setting_file = ''
    data_sub_folder = ''
    for line in lines:
        if line.startswith('#'):
            temp = line.strip()
            if temp in keyword_list:
                keyword = temp
            continue
        if keyword == '#data_keyword':
            data_keyword = line.strip()
        elif keyword == '#data_sub_folder':
            data_sub_folder = line.strip()
        elif keyword == '#data_folder':
            data_folder = line.strip()
        elif keyword == '#attr_num':
            attr_num = int(line.strip())
        elif keyword == '#attr_len':
            attr_len = int(line.strip())
        elif keyword == '#num_classes':
            num_classes = int(line.strip())
        elif keyword == '#start_class':
            start_class = int(line.strip())
        elif keyword == '#class_column':
            class_column = int(line.strip())
        elif keyword == '#log_folder':
            log_folder = line.strip()
        elif keyword == '#obj_folder':
            obj_folder = line.strip()
        elif keyword =='#cnn_setting_file':
            cnn_setting_file = line.strip()
    if attr_num<0 or attr_len<0 or num_classes<0 or start_class<0 or class_column<0:
        raise Exception("Wrong data paramters, please check the parameter file " + parameter_file)

    if check_file_exists(cnn_setting_file) == False:
        raise Exception("Missing cnn setting parameter file: " + cnn_setting_file)
    
    if not log_folder.endswith(split_key):
        log_folder = log_folder + split_key
    log_folder = log_folder + data_keyword + split_key + data_sub_folder + split_key + function_keyword
    #log_folder = log_folder + data_keyword + split_key + function_keyword
    log_folder = init_folder(log_folder)

    if not obj_folder.endswith(split_key):
        obj_folder = obj_folder + split_key
    out_obj_folder = obj_folder + data_keyword + split_key + data_sub_folder + split_key + function_keyword + "_cnn_out_pckl"
    out_model_folder = obj_folder + data_keyword + split_key + data_sub_folder + split_key + function_keyword + "_cnn_temp_saver"
    out_obj_folder = init_folder(out_obj_folder)
    out_model_folder = init_folder(out_model_folder)
    data_folder = data_folder + data_keyword + split_key + data_sub_folder + split_key
    
    return data_keyword, data_folder, attr_num, attr_len, num_classes, start_class, class_column, log_folder, out_obj_folder, out_model_folder, cnn_setting_file



# The function is used to read the input method key file
# The key file contains the data file we need to process
def read_genetic_nn_parameter(parameter_file, function_keyword="genetic_nn"):
    lines = open(parameter_file).readlines()
    split_key = '/'

    keyword_list = ['#num_selected', '#data_keyword', '#data_folder', '#attr_num', '#attr_len', '#num_classes', '#start_class', '#class_column', '#log_folder', '#obj_folder', '#nn_setting_file']

    num_selected = -1
    keyword = ''
    data_keyword = ''
    data_folder = ''
    attr_num = -1
    attr_len = -1
    num_classes = -1
    start_class = -1
    class_column = -1
    log_folder = ''
    obj_folder = ''
    nn_setting_file = ''
    for line in lines:
        if line.startswith('#'):
            temp = line.strip()
            if temp in keyword_list:
                keyword = temp
            continue
        if keyword == '#num_selected':
            num_selected = int(line.strip())
        elif keyword == '#data_keyword':
            data_keyword = line.strip()
        elif keyword == '#data_folder':
            data_folder = line.strip()
        elif keyword == '#attr_num':
            attr_num = int(line.strip())
        elif keyword == '#attr_len':
            attr_len = int(line.strip())
        elif keyword == '#num_classes':
            num_classes = int(line.strip())
        elif keyword == '#start_class':
            start_class = int(line.strip())
        elif keyword == '#class_column':
            class_column = int(line.strip())
        elif keyword == '#log_folder':
            log_folder = line.strip()
        elif keyword == '#obj_folder':
            obj_folder = line.strip()
        elif keyword =='#nn_setting_file':
            nn_setting_file = line.strip()

    if num_selected<0 or attr_num<0 or attr_len<0 or num_classes<0 or start_class<0 or class_column<0:
        raise Exception("Wrong data paramters, please check the parameter file " + parameter_file)

    if check_file_exists(nn_setting_file) == False:
        raise Exception("Missing nn setting parameter file: " + nn_setting_file)
    
    if not log_folder.endswith(split_key):
        log_folder = log_folder + split_key
    log_folder = log_folder + data_keyword + split_key + function_keyword
    log_folder = init_folder(log_folder)

    if not obj_folder.endswith(split_key):
        obj_folder = obj_folder + split_key
    nn_obj_folder = obj_folder + data_keyword + split_key + function_keyword + "_nn_out_pckl"
    print (nn_obj_folder)
    nn_obj_folder = init_folder(nn_obj_folder)
    
    return num_selected, data_keyword, data_folder, attr_num, attr_len, num_classes, start_class, class_column, log_folder, nn_obj_folder, nn_setting_file




# The function is used to read the input method key file
# The key file contains the data file we need to process
def read_nn_classification_parameter(parameter_file, function_keyword="nn_classification"):
    lines = open(parameter_file).readlines()
    split_key = '/'

    keyword_list = ['#data_keyword', '#data_folder', '#attr_num', '#attr_len', '#num_classes', '#start_class', '#class_column', '#log_folder', '#obj_folder', '#nn_setting_file']

    keyword = ''
    data_keyword = ''
    data_folder = ''
    attr_num = -1
    attr_len = -1
    num_classes = -1
    start_class = -1
    class_column = -1
    log_folder = ''
    obj_folder = ''
    nn_setting_file = ''
    for line in lines:
        if line.startswith('#'):
            temp = line.strip()
            if temp in keyword_list:
                keyword = temp
            continue
        if keyword == '#data_keyword':
            data_keyword = line.strip()
        elif keyword == '#data_folder':
            data_folder = line.strip()
        elif keyword == '#attr_num':
            attr_num = int(line.strip())
        elif keyword == '#attr_len':
            attr_len = int(line.strip())
        elif keyword == '#num_classes':
            num_classes = int(line.strip())
        elif keyword == '#start_class':
            start_class = int(line.strip())
        elif keyword == '#class_column':
            class_column = int(line.strip())
        elif keyword == '#log_folder':
            log_folder = line.strip()
        elif keyword == '#obj_folder':
            obj_folder = line.strip()
        elif keyword =='#nn_setting_file':
            nn_setting_file = line.strip()

    if attr_num<0 or attr_len<0 or num_classes<0 or start_class<0 or class_column<0:
        raise Exception("Wrong data paramters, please check the parameter file " + parameter_file)

    if check_file_exists(nn_setting_file) == False:
        raise Exception("Missing nn setting parameter file: " + nn_setting_file)
    
    if not log_folder.endswith(split_key):
        log_folder = log_folder + split_key
    log_folder = log_folder + data_keyword + split_key + function_keyword
    log_folder = init_folder(log_folder)

    if not obj_folder.endswith(split_key):
        obj_folder = obj_folder + split_key
    nn_obj_folder = obj_folder + data_keyword + split_key + "nn_out_pckl"
    print (nn_obj_folder)
    nn_obj_folder = init_folder(nn_obj_folder)
    
    return data_keyword, data_folder, attr_num, attr_len, num_classes, start_class, class_column, log_folder, nn_obj_folder, nn_setting_file



def read_project_feature_generation_parameter(parameter_file, function_keyword="projected_feature_generation"):
    split_key = '/'
    keyword_list = ['#data_keyword', '#attr_num', '#attr_len', '#num_classes', '#start_class', '#class_column', '#method', '#pckl_folder', '#data_folder', '#log_folder', '#out_obj_folder', '#pckl_keyword']
    lines = open(parameter_file).readlines()
    data_keyword = ''
    keyword = ''
    attr_num = -1
    attr_len = -1
    num_classes = -1
    start_class = -1
    class_column = -1
    method = ''
    pckl_folder = ''
    data_folder = ''
    log_folder = ''
    log_postfix = '_projected_feature.log'
    out_obj_folder = ''
    pckl_keyword = ''
    for line in lines:
        if line.startswith('#'):
            temp = line.strip()
            if temp in keyword_list:
                keyword = line.strip()
            continue
        if keyword == '#data_keyword':
            data_keyword = line.strip()
        elif keyword == '#attr_num':
            attr_num = int(line.strip())
        elif keyword == '#attr_len':
            attr_len = int(line.strip())
        elif keyword == '#class_column':
            class_column = int(line.strip())
        elif keyword == '#start_class':
            start_class = int(line.strip())
        elif keyword == '#num_classes':
            num_classes = int(line.strip())
        elif keyword == '#method':
            method = line.strip()
        elif keyword == '#pckl_folder':
            pckl_folder = line.strip()
        elif keyword == '#data_folder':
            data_folder = line.strip()
        elif keyword == '#log_folder':
            log_folder = line.strip()
        elif keyword == '#out_obj_folder':
            out_obj_folder = line.strip()
        elif keyword == '#pckl_keyword':
            pckl_keyword = line.strip()
    if attr_num<0 or attr_len<0 or num_classes<0 or start_class<0 or class_column<0 or check_folder_exists(pckl_folder)==False:
        raise Exception("Wrong data paramters, please check the parameter file " + parameter_file)
    if data_keyword=='' or keyword=='' or method=='' or pckl_folder=='' or data_folder=='' or log_folder=='' or log_postfix=='' or out_obj_folder=='' or pckl_keyword=='':
        raise Exception("Wrong data paramters, please check the parameter file " + parameter_file)

    if not log_folder.endswith(split_key):
        log_folder = log_folder + split_key
    log_folder = log_folder + data_keyword + split_key + function_keyword

    if not out_obj_folder.endswith(split_key):
        out_obj_folder = out_obj_folder + split_key
    out_obj_folder = out_obj_folder + data_keyword + split_key + function_keyword
    log_folder = init_folder(log_folder)
    out_obj_folder = init_folder(out_obj_folder)
        
    return data_keyword, data_folder, attr_num, attr_len, class_column, start_class, num_classes, method, pckl_folder, log_folder, log_postfix, out_obj_folder, pckl_keyword




# The function is used to read the input method key file
# The key file contains the data file we need to process
def read_cnn_varying_classification_parameter(parameter_file):
    lines = open(parameter_file).readlines()
    split_key = '/'
    function_keyword = "cnn_varying_classification"

    keyword_list = ['#data_keyword', '#data_folder', '#attr_num', '#attr_len', '#num_classes', '#start_class', '#class_column', '#method', '#log_folder', '#obj_folder', '#out_model_folder', '#cnn_setting_file']

    keyword = ''
    data_keyword = ''
    data_folder = ''
    attr_num = -1
    attr_len = -1
    num_classes = -1
    start_class = -1
    class_column = -1
    method = ''
    log_folder = ''
    obj_folder = ''
    out_model_folder = ''
    cnn_setting_file = ''
    for line in lines:
        if line.startswith('#'):
            temp = line.strip()
            if temp in keyword_list:
                keyword = temp
            continue

        if keyword == '#data_keyword':
            data_keyword = line.strip()
        elif keyword == '#data_folder':
            data_folder = line.strip()
        elif keyword == '#attr_num':
            attr_num = int(line.strip())
        elif keyword == '#attr_len':
            attr_len = int(line.strip())
        elif keyword == '#num_classes':
            num_classes = int(line.strip())
        elif keyword == '#start_class':
            start_class = int(line.strip())
        elif keyword == '#class_column':
            class_column = int(line.strip())
        elif keyword == '#method':
            method = line.strip()
        elif keyword == '#log_folder':
            log_folder = line.strip()
        elif keyword == '#obj_folder':
            obj_folder = line.strip()
        elif keyword =='#cnn_setting_file':
            cnn_setting_file = line.strip()

    if attr_num<0 or attr_len<0 or num_classes<0 or start_class<0 or class_column<0:
        raise Exception("Wrong data paramters, please check the parameter file " + parameter_file)

    if method == 'cnn' and check_file_exists(cnn_setting_file) == False:
        raise Exception("Missing cnn setting parameter file: " + cnn_setting_file)
    
    if not log_folder.endswith(split_key):
        log_folder = log_folder + split_key
    log_folder = log_folder + data_keyword + split_key + function_keyword
    log_folder = init_folder(log_folder)

    if not obj_folder.endswith(split_key):
        obj_folder = obj_folder + split_key
    out_obj_folder = obj_folder + data_keyword + split_key + "cnn_out_pckl"
    out_model_folder = obj_folder + data_keyword + split_key + "cnn_model_pckl"
    print (out_obj_folder)
    #out_obj_folder = init_folder(out_obj_folder)
    #out_model_folder = init_folder(out_model_folder)
    
    return data_keyword, data_folder, attr_num, attr_len, num_classes, start_class, class_column, method, log_folder, out_obj_folder, out_model_folder, cnn_setting_file




def read_all_feature_classification(parameter_file, function_keyword="all_feature_classification"):
    lines = open(parameter_file).readlines()
    split_key = '/'
    keyword_list = ['#data_keyword', '#data_folder', '#attr_num', '#attr_len', '#num_classes', '#start_class', '#class_column', '#method', '#class_id', '#log_folder', '#obj_folder', '#cnn_setting_file', "#learning"]

    keyword = ''
    data_keyword = ''
    data_folder = ''
    attr_num = -1
    attr_len = -1
    class_id = -1
    num_classes = -1
    start_class = -1
    class_column = -1
    learning_rate = -1
    method = ''
    log_folder = ''
    obj_folder = ''
    obj_file = ''
    out_model_folder = ''
    cnn_setting_file = ''
    for line in lines:
        if line.startswith('#'):
            temp = line.strip()
            if temp in keyword_list:
                keyword = temp
            continue

        if keyword == '#data_keyword':
            data_keyword = line.strip()
        elif keyword == '#data_folder':
            data_folder = line.strip()
        elif keyword == '#attr_num':
            attr_num = int(line.strip())
        elif keyword == '#attr_len':
            attr_len = int(line.strip())
        elif keyword == '#num_classes':
            num_classes = int(line.strip())
        elif keyword == '#start_class':
            start_class = int(line.strip())
        elif keyword == '#class_column':
            class_column = int(line.strip())
        elif keyword == '#class_id':
            class_id = int(line.strip())
        elif keyword == '#method':
            method = line.strip()
        elif keyword == '#log_folder':
            log_folder = line.strip()
        elif keyword == '#obj_folder':
            obj_folder = line.strip()
        elif keyword =='#cnn_setting_file':
            cnn_setting_file = line.strip()
        elif keyword == "#learning":
            learning_rate = float(line.strip())
            
    if attr_num<0 or attr_len<0 or num_classes<0 or start_class<0 or class_column<0:
        raise Exception("Wrong data paramters, please check the parameter file " + parameter_file)

    if method == 'cnn' or method == 'fcn':
        if check_file_exists(cnn_setting_file) == False:
            raise Exception("Missing cnn setting parameter file: " + cnn_setting_file)

    log_folder = log_folder.replace('KEYWORD', data_keyword)
    obj_folder = obj_folder.replace('KEYWORD', data_keyword)

    if not log_folder.endswith(split_key):
        log_folder = log_folder + split_key
    log_folder = log_folder + function_keyword + split_key

    if not obj_folder.endswith(split_key):
        obj_folder = obj_folder + split_key
    obj_folder = obj_folder + function_keyword + split_key

    out_obj_folder = obj_folder + method + "_obj_folder"
    out_model_folder = obj_folder + method + "_model_folder"

    return data_keyword, data_folder, attr_num, attr_len, num_classes, start_class, class_column, class_id, obj_folder, method, log_folder, out_obj_folder, out_model_folder, cnn_setting_file, learning_rate


# The function is used to read the input method key file
# The key file contains the data file we need to process
# function keyword can be "projected_classification" or "global_classification"
def read_pure_feature_generation(parameter_file, function_keyword="pure_feature_generation"):
    lines = open(parameter_file).readlines()
    split_key = '/'
    keyword_list = ['#data_keyword', '#data_folder', '#attr_num', '#attr_len', '#num_classes', '#start_class', '#class_column', '#method', '#log_folder', '#obj_folder']

    keyword = ''
    data_keyword = ''
    data_folder = ''
    attr_num = -1
    attr_len = -1
    class_id = -1
    num_classes = -1
    start_class = -1
    class_column = -1
    method = ''
    log_folder = ''
    obj_folder = ''
    obj_file = ''
    for line in lines:
        if line.startswith('#'):
            temp = line.strip()
            if temp in keyword_list:
                keyword = temp
            continue

        if keyword == '#data_keyword':
            data_keyword = line.strip()
        elif keyword == '#data_folder':
            data_folder = line.strip()
        elif keyword == '#attr_num':
            attr_num = int(line.strip())
        elif keyword == '#attr_len':
            attr_len = int(line.strip())
        elif keyword == '#num_classes':
            num_classes = int(line.strip())
        elif keyword == '#start_class':
            start_class = int(line.strip())
        elif keyword == '#class_column':
            class_column = int(line.strip())
        elif keyword == '#class_id':
            class_id = int(line.strip())
        elif keyword == '#method':
            method = line.strip()
        elif keyword == '#log_folder':
            log_folder = line.strip()
        elif keyword == '#obj_folder':
            obj_folder = line.strip()

    if attr_num<0 or attr_len<0 or num_classes<0 or start_class<0 or class_column<0:
        raise Exception("Wrong data paramters, please check the parameter file " + parameter_file)

    data_folder = data_folder.replace("KEYWORD", data_keyword)
    log_folder = log_folder.replace('KEYWORD', data_keyword)
    obj_folder = obj_folder.replace('KEYWORD', data_keyword)

    if not log_folder.endswith(split_key):
        log_folder = log_folder + split_key
    if not obj_folder.endswith(split_key):
        obj_folder = obj_folder + split_key

    obj_sub_folder = method + "_pure_projected_feature"

    log_folder = log_folder + data_keyword + split_key +function_keyword
    obj_folder = obj_folder + data_keyword + split_key + function_keyword + split_key + obj_sub_folder
    log_folder = init_folder(log_folder)
    obj_folder = init_folder(obj_folder)

    return data_keyword, data_folder, attr_num, attr_len, num_classes, start_class, class_column, class_id, method, log_folder, obj_folder


# The function is used to read the input method key file
# The key file contains the data file we need to process
# function keyword can be "projected_classification" or "global_classification"
def read_cnn_feature_generation(parameter_file, function_keyword="cnn_feature_generation"):
    lines = open(parameter_file).readlines()
    split_key = '/'
    keyword_list = ['#data_keyword', '#data_folder', '#num_classes', '#start_class', '#class_column', '#method', '#log_folder', '#obj_folder']

    keyword = ''
    data_keyword = ''
    data_folder = ''
    attr_num = -1
    attr_len = -1
    class_id = -1
    num_classes = -1
    start_class = -1
    class_column = -1
    top_k = -1
    method = ''
    log_folder = ''
    obj_folder = ''
    obj_file = ''
    out_model_folder = ''
    cnn_setting_file = ''
    for line in lines:
        if line.startswith('#'):
            temp = line.strip()
            if temp in keyword_list:
                keyword = temp
            continue

        if keyword == '#data_keyword':
            data_keyword = line.strip()
        elif keyword == '#data_folder':
            data_folder = line.strip()
        elif keyword == '#attr_num':
            attr_num = int(line.strip())
        elif keyword == '#attr_len':
            attr_len = int(line.strip())
        elif keyword == '#num_classes':
            num_classes = int(line.strip())
        elif keyword == '#start_class':
            start_class = int(line.strip())
        elif keyword == '#class_column':
            class_column = int(line.strip())
        elif keyword == '#class_id':
            class_id = int(line.strip())
        elif keyword == '#top_k':
            top_k = int(line.strip())
        elif keyword == '#method':
            method = line.strip()
        elif keyword == '#log_folder':
            log_folder = line.strip()
        elif keyword == '#obj_folder':
            obj_folder = line.strip()
        elif keyword =='#cnn_setting_file':
            cnn_setting_file = line.strip()

    if attr_num<0 or attr_len<0 or num_classes<0 or start_class<0 or class_column<0 or top_k<0:
        raise Exception("Wrong data paramters, please check the parameter file " + parameter_file)

    if method == 'cnn' and check_file_exists(cnn_setting_file) == False:
        raise Exception("Missing cnn setting parameter file: " + cnn_setting_file)
    
    log_folder = log_folder.replace('KEYWORD', data_keyword)
    obj_folder = obj_folder.replace('KEYWORD', data_keyword)

    if not obj_folder.endswith(split_key):
        out_obj_folder = obj_folder + "_cnn_out_pckl"
        out_model_folder = obj_folder +  "_cnn_temp_saver"
        obj_folder = obj_folder + split_key
    else:
        out_obj_folder = obj_folder[:-1] + "_cnn_out_pckl"
        out_model_folder = obj_folder[:-1] +  "_cnn_temp_saver"
    return data_keyword, data_folder, attr_num, attr_len, num_classes, start_class, class_column, class_id, obj_folder, top_k, method, log_folder, out_obj_folder, out_model_folder, cnn_setting_file


# The function is used to read the input method key file
# The key file contains the data file we need to process
# function keyword can be "projected_classification" or "global_classification"
def read_feature_classification(parameter_file, function_keyword="projected_classification"):
    lines = open(parameter_file).readlines()
    split_key = '/'
    keyword_list = ['#data_keyword', '#data_folder', '#attr_num', '#attr_len', '#num_classes', '#start_class', '#class_column', '#top_k', '#method', '#class_id', '#log_folder', '#obj_folder', '#obj_keyword', '#cnn_setting_file']

    keyword = ''
    data_keyword = ''
    data_folder = ''
    attr_num = -1
    attr_len = -1
    class_id = -1
    num_classes = -1
    start_class = -1
    class_column = -1
    top_k = -1
    method = ''
    log_folder = ''
    obj_folder = ''
    obj_keyword = ''
    out_model_folder = ''
    cnn_setting_file = ''
    for line in lines:
        if line.startswith('#'):
            temp = line.strip()
            if temp in keyword_list:
                keyword = temp
            continue

        if keyword == '#data_keyword':
            data_keyword = line.strip()
        elif keyword == '#data_folder':
            data_folder = line.strip()
        elif keyword == '#attr_num':
            attr_num = int(line.strip())
        elif keyword == '#attr_len':
            attr_len = int(line.strip())
        elif keyword == '#num_classes':
            num_classes = int(line.strip())
        elif keyword == '#start_class':
            start_class = int(line.strip())
        elif keyword == '#class_column':
            class_column = int(line.strip())
        elif keyword == '#class_id':
            class_id = int(line.strip())
        elif keyword == '#top_k':
            top_k = int(line.strip())
        elif keyword == '#method':
            method = line.strip()
        elif keyword == '#log_folder':
            log_folder = line.strip()
        elif keyword == '#obj_folder':
            obj_folder = line.strip()
        elif keyword == '#obj_keyword':
            obj_keyword = line.strip()
        elif keyword =='#cnn_setting_file':
            cnn_setting_file = line.strip()

    if attr_num<0 or attr_len<0 or num_classes<0 or start_class<0 or class_column<0 or top_k<0:
        raise Exception("Wrong data paramters, please check the parameter file " + parameter_file)


    if method == 'cnn' and check_file_exists(cnn_setting_file) == False:
        raise Exception("Missing cnn setting parameter file: " + cnn_setting_file)
    
    log_folder = log_folder.replace('KEYWORD', data_keyword)
    log_folder = log_folder + function_keyword + split_key + obj_keyword
    obj_folder = obj_folder.replace('KEYWORD', data_keyword)
    
    if not obj_folder.endswith(split_key):
        obj_folder = obj_folder + split_key
    
    obj_folder = obj_folder + obj_keyword + split_key
    output_folder = "../../object/" + data_keyword + split_key + function_keyword + split_key + obj_keyword
    
    out_obj_folder = output_folder +  "_top" + str(top_k) + "_cnn_obj_folder" + split_key
    out_model_folder = output_folder + "_top" + str(top_k) + "_cnn_model_folder" + split_key
    
    return data_keyword, data_folder, attr_num, attr_len, num_classes, start_class, class_column, class_id, obj_folder, top_k, method, log_folder, out_obj_folder, out_model_folder, cnn_setting_file


# The function is used to read the input method key file
# The key file contains the data file we need to process
def read_grouped_projected_classification(parameter_file, function_keyword="grouped_projected_classification"):
    lines = open(parameter_file).readlines()
    split_key = '/'
    keyword_list = ['#data_keyword', '#data_folder', '#data_sub_folder', '#attr_num', '#attr_len', '#num_classes', '#start_class', '#class_column', '#method', '#log_folder', '#obj_folder', '#obj_sub_keyword', '#cnn_setting_file']

    keyword = ''
    data_keyword = ''
    data_folder = ''
    data_sub_folder = ''
    attr_num = -1
    attr_len = -1
    start_class = -1
    class_column = -1
    method = ''
    log_folder = ''
    obj_folder = ''
    obj_sub_keyword = ''
    cnn_setting_file = ''
    for line in lines:
        if line.startswith('#'):
            temp = line.strip()
            if temp in keyword_list:
                keyword = temp
            continue

        if keyword == '#data_keyword':
            data_keyword = line.strip()
        elif keyword == '#data_folder':
            data_folder = line.strip()
        elif keyword == '#data_sub_folder':
            data_sub_folder = line.strip()
        elif keyword == '#attr_num':
            attr_num = int(line.strip())
        elif keyword == '#attr_len':
            attr_len = int(line.strip())
        elif keyword == '#start_class':
            start_class = int(line.strip())
        elif keyword == '#class_column':
            class_column = int(line.strip())
        elif keyword == '#method':
            method = line.strip()
        elif keyword == '#log_folder':
            log_folder = line.strip()
        elif keyword == '#obj_folder':
            obj_folder = line.strip()
        elif keyword == '#obj_sub_keyword':
            obj_sub_keyword = line.strip()
        elif keyword =='#cnn_setting_file':
            cnn_setting_file = line.strip()

    if attr_num<0 or attr_len<0 or start_class<0 or class_column<0:
        raise Exception("Wrong data paramters, please check the parameter file " + parameter_file)

    if method == 'cnn' and check_file_exists(cnn_setting_file) == False:
        raise Exception("Missing cnn setting parameter file: " + cnn_setting_file)
    
    if not data_folder.endswith(split_key):
        data_folder = data_folder + split_key
    data_folder = data_folder + data_sub_folder + split_key

    if not log_folder.endswith(split_key):
        log_folder = log_folder + split_key
    log_folder = log_folder + data_keyword + split_key + data_sub_folder + split_key + function_keyword
    #log_folder = init_folder(log_folder)

    if not obj_folder.endswith(split_key):
        obj_folder = obj_folder + split_key
    input_obj_folder = obj_folder + data_keyword + split_key + data_sub_folder + split_key
    output_obj_folder = obj_folder + data_keyword + split_key + data_sub_folder + split_key + function_keyword
    #output_obj_folder = init_folder(output_obj_folder)
    return data_folder, attr_num, attr_len, start_class, class_column, input_obj_folder, obj_sub_keyword, output_obj_folder, method, log_folder, cnn_setting_file



#def read_global_feature_classification(parameter_file, function_keyword = "global_classification"):
#    split_key = '/'
#    keyword_list = ['#data_keyword', '#data_folder', '#attr_num', '#attr_len', '#num_classes', '#start_class', '#class_column', '#top_attr', #'#feature_obj_folder', '#method', '#log_folder', '#cnn_setting_file']
#    
#    lines = open(parameter_file).readlines()
#    data_keyword = ''
#    data_folder = ''
#    attr_num = -1
#    attr_len = -1
#    num_classes = -1
#    start_class = -1
#    class_column = -1
#    method = ''
#    top_attr = ''
#    feature_obj_folder = ''
#    log_folder = ''
#    feature_keyword = ''
#    cnn_setting_file = ''
#    for line in lines:
#        if line.startswith('#'):
#            temp = line.strip()
#            if temp in keyword_list:
#                keyword = line.strip()
#            continue
#        if keyword == '#data_keyword':
#            data_keyword = line.strip()
#        elif keyword == '#data_folder':
#            data_folder = line.strip()
#        elif keyword == '#attr_num':
#            attr_num = int(line.strip())
#        elif keyword == '#attr_len':
#            attr_len = int(line.strip())
#        elif keyword == '#class_column':
#            class_column = int(line.strip())
#        elif keyword == '#start_class':
#            start_class = int(line.strip())
#        elif keyword == '#num_classes':
#            num_classes = int(line.strip())
#        elif keyword == '#top_attr':
#            top_attr = int(line.strip())
#        elif keyword == '#feature_obj_folder':
#            feature_obj_folder = line.strip()
#        elif keyword == '#method':
#            method = line.strip()
#        elif keyword == '#log_folder':
#            log_folder = line.strip()
#        elif keyword == '#cnn_setting_file':
#            cnn_setting_file = line.strip()
#    if attr_num<0 or attr_len<0 or num_classes<0 or start_class<0 or class_column<0 or top_attr<0:
#        raise Exception("Wrong data paramters, please check the parameter file " + parameter_file)
#
#    if method == 'cnn' and check_file_exists(cnn_setting_file) == False:
#        raise Exception("Missing cnn setting parameter file: " + cnn_setting_file)
#
#    feature_keyword = feature_obj_file.split(split_key)[-1]
#    feature_keyword = feature_keyword[0:feature_keyword.index('_global_feature.pckl')]
#    
#    if not log_folder.endswith(split_key):
#        log_folder = log_folder + split_key
#    log_folder = log_folder + data_keyword + split_key + function_keyword
#    log_folder = init_folder(log_folder)
#
#    return data_keyword, data_folder, attr_num, attr_len, num_classes, start_class, class_column, top_attr, feature_obj_file, method, log_folder, feature_keyword, cnn_setting_file



if __name__ == '__main__':
    #parameter_file = '../../parameters/global_feature_generation.txt'
    #read_global_feature_generation_parameter(parameter_file)

    parameter_file = '../../parameters/global_feature_classification.txt'
    read_feature_classification(parameter_file)




# The function is used to read the input data key file
# The key file contains the data file we need to process
# The file line is the keyword for this data folder
# The second line is the directory to this data folder
#def read_data_key_file(input_key_file):
#    keyword_list = ['#data_keyword', '#data_folder', '#attr_num', '#attr_len', '#num_classes', '#start_class', '#class_column']
#    lines = open(input_key_file).readlines()
#
#    keyword = ''
#    data_keyword = ''
#    data_folder = ''
#    attr_num = -1
#    attr_len = -1
#    class_column = -1
#    start_class = -1
#    num_classes = -1
#    for line in lines:
#        if line.startswith('#'):
#            temp = line.strip()
#            if temp in keyword_list:
#                keyword = temp
#            continue
#
#        if keyword == '#data_key':
#            data_key = line.strip()
#        elif keyword == '#data_folder':
#            data_folder = line.strip()
#        elif keyword == '#attr_num':
#            attr_num = int(line.strip())
#        elif keyword == '#attr_len':
#            attr_len = int(line.strip())
#        elif keyword == '#class_column':
#            class_column = int(line.strip())
#        elif keyword == '#start_class':
#            start_class = int(line.strip())
#        elif keyword == '#num_classes':
#            num_classes = int(line.strip())
#
#    return data_keyword, data_folder, attr_num, attr_len, num_classes, class_column, start_class
#
#
## The function is used to read the input method key file
## The key file contains the data file we need to process
## The file line is the keyword for this data folder
## The second line is the directory to this data folder
#def read_method_key_file(method_key_file):
#    method = ""
#    lines = open(method_key_file).readlines()
#    method = lines[0].strip()
#    log_folder = lines[1].strip()
#    save_obj_folder = lines[2].strip()
#
#    return method, log_folder, save_obj_folder
#
#
#def read_project_feature_generation_parameter(parameter_file):
#    function_keyword = 'projected_feature'
#    split_key = '/'
#    keyword_list = ['#data_keyword', '#attr_num', '#attr_len', '#num_classes', '#start_class', '#class_column', '#method', '#pckl_folder', '#data_folder', '#log_folder', '#out_obj_folder', '#pckl_keyword']
#    lines = open(parameter_file).readlines()
#    data_keyword = ''
#    keyword = ''
#    attr_num = -1
#    attr_len = -1
#    num_classes = -1
#    start_class = -1
#    class_column = -1
#    method = ''
#    pckl_folder = ''
#    data_folder = ''
#    log_folder = ''
#    log_postfix = '_projected_feature.log'
#    out_obj_folder = ''
#    pckl_keyword = ''
#    for line in lines:
#        if line.startswith('#'):
#            temp = line.strip()
#            if temp in keyword_list:
#                keyword = line.strip()
#            continue
#        if keyword == '#data_keyword':
#            data_keyword = line.strip()
#        elif keyword == '#attr_num':
#            attr_num = int(line.strip())
#        elif keyword == '#attr_len':
#            attr_len = int(line.strip())
#        elif keyword == '#class_column':
#            class_column = int(line.strip())
#        elif keyword == '#start_class':
#            start_class = int(line.strip())
#        elif keyword == '#num_classes':
#            num_classes = int(line.strip())
#        elif keyword == '#method':
#            method = line.strip()
#        elif keyword == '#pckl_folder':
#            pckl_folder = line.strip()
#        elif keyword == '#data_folder':
#            data_folder = line.strip()
#        elif keyword == '#log_folder':
#            log_folder = line.strip()
#        elif keyword == '#out_obj_folder':
#            out_obj_folder = line.strip()
#        elif keyword == '#pckl_keyword':
#            pckl_keyword = line.strip()
#    if attr_num<0 or attr_len<0 or num_classes<0 or start_class<0 or class_column<0 or check_folder_exists(pckl_folder)==False:
#        raise Exception("Wrong data paramters, please check the parameter file " + parameter_file)
#    if data_keyword=='' or keyword=='' or method=='' or pckl_folder=='' or data_folder=='' or log_folder=='' or log_postfix=='' or out_obj_folder=='' or pckl_keyword=='':
#        raise Exception("Wrong data paramters, please check the parameter file " + parameter_file)
#
#    if not log_folder.endswith(split_key):
#        log_folder = log_folder + split_key
#    log_folder = log_folder + data_keyword + split_key + function_keyword
#
#    if not out_obj_folder.endswith(split_key):
#        out_obj_folder = out_obj_folder + split_key
#    out_obj_folder = out_obj_folder + data_keyword + split_key + function_keyword
#    log_folder = init_folder(log_folder)
#    out_obj_folder = init_folder(out_obj_folder)
#        
#    return data_keyword, data_folder, attr_num, attr_len, class_column, start_class, num_classes, method, pckl_folder, log_folder, log_postfix, out_obj_folder, pckl_keyword
#
#def read_global_feature_generation_parameter(parameter_file):
#    split_key = '/'
#    global_feature_keyword = 'global_feature_generation'
#    log_postfix = '_' + global_feature_keyword + '.log'
#    keyword_list = ['#data_keyword', '#data_folder', '#attr_num', '#attr_len', '#num_classes', '#start_class', '#class_column', '#pckl_folder', '#log_folder', '#log_postfix', '#save_obj_folder']
#    lines = open(parameter_file).readlines()
#    data_keyword = ''
#    data_folder = ''
#    attr_num = -1
#    attr_len = -1
#    num_classes = -1
#    start_class = -1
#    class_column = -1
#    #method = ''
#    pckl_folder = ''
#    log_folder = ''
#    for line in lines:
#        if line.startswith('#'):
#            temp = line.strip()
#            if temp in keyword_list:
#                keyword = line.strip()
#            continue
#        if keyword == '#data_keyword':
#            data_keyword = line.strip()
#        elif keyword == '#data_folder':
#            data_folder = line.strip()
#        elif keyword == '#attr_num':
#            attr_num = int(line.strip())
#        elif keyword == '#attr_len':
#            attr_len = int(line.strip())
#        elif keyword == '#class_column':
#            class_column = int(line.strip())
#        elif keyword == '#start_class':
#            start_class = int(line.strip())
#        elif keyword == '#num_classes':
#            num_classes = int(line.strip())
#        #elif keyword == '#method':
#        #    method = line.strip()
#        elif keyword == '#pckl_folder':
#            pckl_folder = line.strip()
#        elif keyword == '#log_folder':
#            log_folder = line.strip()
#
#    if attr_num<0 or attr_len<0 or num_classes<0 or start_class<0 or class_column<0 or check_folder_exists(pckl_folder)==False:
#        raise Exception("Wrong data paramters, please check the parameter file " + parameter_file)
#
#    if not log_folder.endswith(split_key):
#        log_folder = log_folder + split_key
#    log_folder = log_folder + data_keyword + split_key + global_feature_keyword
#    out_obj_folder = pckl_folder.split(data_keyword)[0] + data_keyword + split_key + global_feature_keyword
#    log_folder = init_folder(log_folder)
#    out_obj_folder = init_folder(out_obj_folder)
#
#    return data_keyword, data_folder, attr_num, attr_len, class_column, start_class, num_classes, pckl_folder, log_folder, log_postfix, out_obj_folder
#
#
## Read the parameter file to transfet between projected and global features
#def read_pure_projected_feature_gene_parameter(parameter_file):
#    split_key = '/'
#    function_keyword = 'pure_projected_feature_generation'
#
#    keyword_list = ['#data_keyword', '#data_folder', '#class_column', '#attr_num', '#num_classes', '#method', '#transpose', '#log_folder', '#pckl_folder']
#    
#    lines = open(parameter_file).readlines()
#    data_keyword = ''
#    data_folder = ''
#    class_column = -1
#    attr_num = -1
#    num_classes = -1
#    method = ''
#    transpose = ''
#    pckl_folder = ''
#    log_folder = ''
#    keyword = ''
#    transpose = ''
#
#    for line in lines:
#        if line.startswith('#'):
#            temp = line.strip()
#            if temp in keyword_list:
#                keyword = line.strip()
#            continue
#        if keyword == '#data_keyword':
#            data_keyword = line.strip()
#        elif keyword == '#data_folder':
#            data_folder = line.strip()
#        elif keyword == '#class_column':
#            class_column = int(line.strip())
#        elif keyword == '#attr_num':
#            attr_num = int(line.strip())
#        elif keyword == '#num_classes':
#            num_classes = int(line.strip())
#        elif keyword == '#method':
#            method = line.strip()
#        elif keyword == '#transpose':
#            transpose = line.strip()
#        elif keyword == '#pckl_folder':
#            pckl_folder = line.strip()
#        elif keyword == '#log_folder':
#            log_folder = line.strip()
#
#    if check_folder_exists(pckl_folder)==False or check_folder_exists(log_folder)==False:
#        raise Exception("Wrong data paramters, please check the parameter file " + parameter_file)
#
#    if not log_folder.endswith(split_key):
#        log_folder = log_folder + split_key
#    log_folder = log_folder + data_keyword + split_key + function_keyword
#    log_folder = init_folder(log_folder)
#    log_file = log_folder + method
#
#    if not pckl_folder.endswith(split_key):
#        pckl_folder = pckl_folder + split_key
#    pckl_folder = pckl_folder + data_keyword + split_key + function_keyword
#    pckl_folder = init_folder(pckl_folder)
#    out_obj_file = pckl_folder + method + '.pckl'
#
#    if transpose == 'False':
#        transpose = False
#    else:
#        transpose = True
#
#    return data_folder, class_column, attr_num, num_classes, data_keyword, method, transpose, log_file, out_obj_file
#
#
#
#
## Read the parameter file to transfet between projected and global features
#def read_projected_global_parameter(parameter_file):
#    split_key = '/'
#
#    keyword_list = ['#data_keyword', '#num_classes', '#projected_obj_folder', '#global_obj_folder', '#pckl_folder', '#log_folder', '#input_obj_file', '#projected_to_global']
#    lines = open(parameter_file).readlines()
#    data_keyword = ''
#    num_classes = -1
#    projected_obj_folder = ''
#    global_obj_folder = ''
#    pckl_folder = ''
#    input_obj_file = ''
#    projected_to_global = None
#    log_folder = ''
#
#    for line in lines:
#        if line.startswith('#'):
#            temp = line.strip()
#            if temp in keyword_list:
#                keyword = line.strip()
#            continue
#        if keyword == '#data_keyword':
#            data_keyword = line.strip()
#        elif keyword == '#projected_obj_folder':
#            projected_obj_folder = line.strip()
#        elif keyword == '#global_obj_folder':
#            global_obj_folder = line.strip()
#        elif keyword == '#num_classes':
#            num_classes = int(line.strip())
#        elif keyword == '#input_obj_file':
#            input_obj_file = line.strip()
#        elif keyword == '#projected_to_global':
#            projected_to_global = line.strip()
#        elif keyword == '#pckl_folder':
#            pckl_folder = line.strip()
#        elif keyword == '#log_folder':
#            log_folder = line.strip()
#
#    if check_folder_exists(pckl_folder)==False:
#        raise Exception("Wrong data paramters, please check the parameter file " + parameter_file)
#
#    if projected_to_global == '':
#        raise Exception("projected_to_global parameter is missing")
#
#    if not pckl_folder.endswith(split_key):
#        pckl_folder = pckl_folder + split_key
#    projected_obj_folder = pckl_folder + data_keyword + split_key + projected_obj_folder
#    global_obj_folder = pckl_folder + data_keyword + split_key + global_obj_folder
#
#    projected_obj_folder = init_folder(projected_obj_folder)
#    global_obj_folder = init_folder(global_obj_folder)
#
#
#    if projected_to_global == "True":
#        projected_obj_file = input_obj_file
#        global_obj_file = projected_obj_file.replace('projected', 'global')
#        function_keyword = "projected_to_global"
#    elif projected_to_global == "False":
#        global_obj_file = input_obj_file
#        projected_obj_file = global_obj_file.replace('global', 'projected')
#        projected_obj_folder = init_folder(projected_obj_folder)
#        function_keyword = "global_to_projected"
#    elif projected_to_global == "All":
#        projected_obj_file = input_obj_file
#        global_obj_file = projected_obj_file.replace('projected', 'global')
#        global_obj_folder = init_folder(global_obj_folder)
#        function_keyword = "projected_to_all"
#
#
#
#    if not log_folder.endswith(split_key):
#        log_folder = log_folder + split_key
#    log_folder = log_folder + data_keyword + split_key + 'projected_global_tran'
#    log_folder = init_folder(log_folder)
#
#    log_file = log_folder + function_keyword
#
#    return data_keyword, num_classes, projected_obj_folder, projected_obj_file, global_obj_folder, global_obj_file, log_file, projected_to_global
#
#
#
#
## The function is used to read the input method key file
## The key file contains the data file we need to process
## The file line is the keyword for this data folder
## The second line is the directory to this data folder
#def read_project_feature_classification_parameter(parameter_file):
#    split_key = '/'
#    function_keyword = "projected_classification_10_1_fold"
#    keyword_list = ['#data_keyword', '#data_folder', '#file_keyword', '#attr_num', '#attr_len', '#num_classes', '#start_class', '#class_column', '#top_attr', '#feature_obj_folder', '#feature_obj_file', '#method', '#log_folder', '#cnn_setting_file']
#    
#    lines = open(parameter_file).readlines()
#    keyword = ''
#    data_keyword = ''
#    file_keyword = ''
#    data_folder = ''
#    attr_num = -1
#    attr_len = -1
#    num_classes = -1
#    start_class = -1
#    class_column = -1
#    method = ''
#    top_attr = ''
#    feature_obj_file = ''
#    feature_obj_folder = ''
#    log_folder = ''
#    feature_keyword = ''
#    cnn_setting_file = ''
#    for line in lines:
#        line = line.strip()
#        if line.startswith('#') or len(line)==0:
#            if line in keyword_list:
#                keyword = line
#            continue
#        if keyword == '#data_keyword':
#            data_keyword = line
#        elif keyword == '#data_folder':
#            data_folder = line
#        elif keyword == '#file_keyword':
#            file_keyword = line
#        elif keyword == '#attr_num':
#            attr_num = int(line)
#        elif keyword == '#attr_len':
#            attr_len = int(line)
#        elif keyword == '#class_column':
#            class_column = int(line)
#        elif keyword == '#start_class':
#            start_class = int(line)
#        elif keyword == '#num_classes':
#            num_classes = int(line)
#        elif keyword == '#top_attr':
#            top_attr = int(line)
#        elif keyword == '#feature_obj_file':
#            feature_obj_file = line
#        elif keyword == '#feature_obj_folder':
#            feature_obj_folder = line
#        elif keyword == '#method':
#            method = line
#        elif keyword == '#log_folder':
#            log_folder = line
#        elif keyword == '#cnn_setting_file':
#            cnn_setting_file = line
#    if not feature_obj_folder.endswith(split_key):
#        feature_obj_folder = feature_obj_folder + split_key 
#    if attr_num<0 or attr_len<0 or num_classes<0 or start_class<0 or class_column<0 or top_attr<0:
#        print attr_num
#        print attr_len
#        print num_classes
#        print start_class
#        print class_column
#        print top_attr
#        print feature_obj_folder+feature_obj_file
#        #print check_file_exists(feature_obj_folder+feature_obj_file)
#        raise Exception("Wrong data paramters, please check the parameter file " + parameter_file)
#
#    if method == 'cnn' and check_file_exists(cnn_setting_file) == False:
#        raise Exception("Missing cnn setting parameter file: " + cnn_setting_file)
#
#    print feature_obj_file
#    feature_keyword = feature_obj_file.split(split_key)[-1]
#    feature_keyword = feature_keyword[0:feature_keyword.index('_projected_feature.pckl')]
#    
#    if not log_folder.endswith(split_key):
#        log_folder = log_folder + split_key
#    log_folder = log_folder + data_keyword + split_key + function_keyword
#    log_folder = init_folder(log_folder)
#    #feature_obj_folder = init_folder(feature_obj_folder)
#    
#    temp_obj_folder = feature_obj_folder + feature_obj_file + '_' + method + "_temp/"
#    init_folder(temp_obj_folder)
#
#    return data_keyword, data_folder, file_keyword, attr_num, attr_len, num_classes, start_class, class_column, top_attr, feature_obj_folder, feature_obj_file, method, log_folder, feature_keyword, cnn_setting_file, temp_obj_folder
#
#
#def read_projected_feature_load_prediction_parameter(parameter_file):
#    split_key = '/'
#    function_keyword = "projected_load_prediction"
#    keyword_list = ['#data_keyword', '#data_folder', '#file_keyword', '#attr_num', '#attr_len', '#num_classes', '#start_class', '#class_column', '#top_attr', '#feature_obj_folder', '#feature_obj_file', '#feature_col_update', '#log_folder', '#cnn_setting_file', '#cnn_pckl_folder', '#cnn_pckl_keyword']
#    
#    lines = open(parameter_file).readlines()
#    keyword = ''
#    data_keyword = ''
#    file_keyword = ''
#    data_folder = ''
#    attr_num = -1
#    attr_len = -1
#    num_classes = -1
#    start_class = -1
#    class_column = -1
#    top_attr = ''
#    feature_obj_file = ''
#    feature_obj_folder = ''
#    log_folder = ''
#    feature_keyword = ''
#    cnn_setting_file = ''
#    cnn_pckl_folder = ''
#    cnn_pckl_keyword = ''
#    feature_col_update = 'True'
#
#    for line in lines:
#        line = line.strip()
#        if line.startswith('#') or len(line)==0:
#            if line in keyword_list:
#                keyword = line
#            continue
#        if keyword == '#data_keyword':
#            data_keyword = line
#        elif keyword == '#data_folder':
#            data_folder = line
#        elif keyword == '#file_keyword':
#            file_keyword = line
#        elif keyword == '#attr_num':
#            attr_num = int(line)
#        elif keyword == '#attr_len':
#            attr_len = int(line)
#        elif keyword == '#class_column':
#            class_column = int(line)
#        elif keyword == '#start_class':
#            start_class = int(line)
#        elif keyword == '#num_classes':
#            num_classes = int(line)
#        elif keyword == '#top_attr':
#            top_attr = int(line)
#        elif keyword == '#feature_obj_file':
#            feature_obj_file = line
#        elif keyword == '#feature_obj_folder':
#            feature_obj_folder = line
#        elif keyword == '#feature_col_update':
#            feature_col_update = line
#        elif keyword == '#log_folder':
#            log_folder = line
#        elif keyword == '#cnn_pckl_folder':
#            cnn_pckl_folder = line
#        elif keyword == '#cnn_pckl_keyword':
#            cnn_pckl_keyword = line
#        elif keyword == '#cnn_setting_file':
#            cnn_setting_file = line
#    if not feature_obj_folder.endswith(split_key):
#        feature_obj_folder = feature_obj_folder + split_key 
#    if attr_num<0 or attr_len<0 or num_classes<0 or start_class<0 or class_column<0 or top_attr<0:
#        print attr_num
#        print attr_len
#        print num_classes
#        print start_class
#        print class_column
#        print top_attr
#        print feature_obj_folder+feature_obj_file
#        #print check_file_exists(feature_obj_folder+feature_obj_file)
#        raise Exception("Wrong data paramters, please check the parameter file " + parameter_file)
#
#    print feature_obj_file
#    feature_keyword = feature_obj_file.split(split_key)[-1]
#    feature_keyword = feature_keyword[0:feature_keyword.index('_projected_feature.pckl')]
#    
#    if not log_folder.endswith(split_key):
#        log_folder = log_folder + split_key
#    log_folder = log_folder + data_keyword + split_key + function_keyword
#    log_folder = init_folder(log_folder)
#    #feature_obj_folder = init_folder(feature_obj_folder)
#    #temp_obj_folder = feature_obj_folder + feature_obj_file + "_temp/"
#    #init_folder(temp_obj_folder)
#
#    if feature_col_update == 'False':
#        feature_col_update = False
#    else:
#        feature_col_update = True
#
#    return data_keyword, data_folder, file_keyword, attr_num, attr_len, num_classes, start_class, class_column, top_attr, feature_obj_folder, feature_obj_file, feature_col_update, log_folder, feature_keyword, cnn_pckl_folder, cnn_pckl_keyword, cnn_setting_file
#
#
#
## The function is used to read the input method key file
## The key file contains the data file we need to process
## The file line is the keyword for this data folder
## The second line is the directory to this data folder


