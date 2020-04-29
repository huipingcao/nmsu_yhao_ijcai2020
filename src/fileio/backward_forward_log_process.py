import logging
import numpy as np
from data_io import list_files
from data_io import init_folder
from data_io import write_to_excel
from object_io import save_obj
import os


def results_from_file(file_name, line_keyword):
    feature_dict = {}
    min_class = 100
    max_class = -1
    with open(file_name) as f:
        value_vector = []
        for line in f:
            if line_keyword in line:
                line_array = line.split(':')
                out_string = line_array[-1].strip()
                class_str = line_array[-2]
                class_str = class_str.replace(line_keyword, "")
                class_id = int(class_str)
                print class_id
                if min_class > class_id:
                    min_class = class_id
                if max_class < class_id:
                    max_class = class_id
                feature_str = out_string.replace('[', '').replace(']', '')
                feature_str = feature_str.split(" ")
                feature_vector = []
                for item in feature_str:
                    try:
                        feature_vector.append(int(item))
                    except Exception:
                        continue
                    
                print feature_vector

                if class_id not in feature_dict.keys():
                    feature_dict[class_id] = feature_vector
                else:
                    raise Exception("Class " + str(class_id) + " found more than once!!!")
            else:
                continue
    max_class = max_class + 1
    feature_matrix = []
    for i in range(min_class, max_class):
        if i not in feature_dict.keys():
            raise Exception("Class " + str(class_id) + " missing!!!")
        else:
            feature_matrix.append(feature_dict[i])
    feature_matrix = np.array(feature_matrix)
    return feature_matrix


def results_from_folder(folder_name, out_obj_folder, file_keyword, num_classes, line_keyword):
    file_list = list_files(folder_name)
    file_count = 0
    for file_name in file_list:
        if file_name.startswith('.'):
            continue
        if file_keyword not in file_name:
            continue
        print file_name
        file_count = file_count + 1
        feature_matrix = results_from_file(folder_name+file_name, line_keyword)
        print feature_matrix.shape
        out_obj_file = file_name.split('.')[0] + "_top15.out"
        save_obj([feature_matrix], out_obj_folder + out_obj_file)



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
    method = "rf"

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
    folder_keyword = "forward_wrapper"
    folder_name = log_folder + folder_keyword + "/"
    out_obj_folder = "../../object/" + data_key + "/" + folder_keyword
    out_obj_folder = init_folder(out_obj_folder)
    file_keyword = ""

    line_keyword = 'Top Features For Class '
    
    results_from_folder(folder_name, out_obj_folder, file_keyword, num_classes, line_keyword)
    
