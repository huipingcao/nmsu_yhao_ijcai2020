import numpy as np
from sklearn.metrics import accuracy_score
from fileio.log_io import init_logging


def softmax(x_vector):
    e_x = np.exp(x_vector - np.max(x_vector))
    return e_x/e_x.sum()


def softmax_2d(x_2d):
    x_len, y_len = x_2d.shape
    ret_x = []
    for i in range(x_len):
        x_v = x_2d[i, :]
        e_x = np.exp(x_v - np.max(x_v))
        ret_x.append(e_x/e_x.sum())
    return np.array(ret_x)



def result_analysis(predict_y_proba, test_y_vector, logger=None):
    if logger is None:
        logger = init_logging('')
    test_len = len(test_y_vector)
    if len(predict_y_proba.shape) == 2:
        predict_y_vector = np.argmax(predict_y_proba, axis=1)
    else:
        predict_y_vector = predict_y_proba
    min_class = min(test_y_vector)
    max_class = max(test_y_vector) + 1
    for c in range(min_class, max_class):
        logger.info("Class: " + str(c))
        print("Class: " + str(c))
        for i in range(test_len):
            real_c = test_y_vector[i]
            pred_c = predict_y_vector[i]
            if real_c == c:
                if real_c != pred_c:
                    logger.info("True label: " + str(real_c))
                    logger.info("Predict label: " + str(pred_c))
                    print("ins: " + str(i))
                    print("True label: " + str(real_c))
                    print("Predict label: " + str(pred_c))
                    if len(predict_y_proba.shape) == 2:
                        logger.info(predict_y_proba[i, :])

def result_analysis_with_x(predict_y_proba, test_y_vector, test_x_matrix, logger=None):
    if logger is None:
        logger = init_logging('')
    test_len = len(test_y_vector)
    if len(predict_y_proba.shape) == 2:
        predict_y_vector = np.argmax(predict_y_proba, axis=1)
    else:
        predict_y_vector = predict_y_proba
    min_class = min(test_y_vector)
    max_class = max(test_y_vector) + 1
    #for c in range(min_class, max_class):
    for c in [1, 7]:
        logger.info("Class: " + str(c))
        for i in range(test_len):
            real_c = test_y_vector[i]
            pred_c = predict_y_vector[i]
            if real_c == c:
                if real_c != pred_c:
                    logger.info("True label: " + str(real_c))
                    logger.info("Predict label: " + str(pred_c))
                    logger.info(test_x_matrix[i, :])
                    x_v = test_x_matrix[i, :]
                    #logger.info(np.exp(test_x_matrix[i, :] - np.max(test_x_matrix[i, :])))
                    logger.info(x_v - min(x_v))
                    #if len(predict_y_proba.shape) == 2:
                    #    logger.info(predict_y_proba[i, :])


def averaged_class_based_accuracy(predict_y_vector, real_y_vector):
    min_class = min(real_y_vector)
    max_class = max(real_y_vector) + 1

    averaged_accuracy = 0
    ret_str = "class based accuracy: "
    for c in range(min_class, max_class):
        class_index = np.where(real_y_vector==c)[0]
        class_predict_y = predict_y_vector[class_index]
        class_real_y = real_y_vector[class_index]
        class_accuracy = accuracy_score(class_real_y, class_predict_y, True)
        values, counts = np.unique(class_predict_y, return_counts=True)
        all_count = len(class_index)
        ret_str = ret_str + str(c) + ":" + str(class_accuracy) + " ALL:" + str(all_count) + " ["
        for i in range(len(values)):
            ret_str = ret_str + str(values[i]) + ":" + str(counts[i]) + ","
        ret_str = ret_str + "]\n"
        averaged_accuracy = averaged_accuracy + class_accuracy
        
    return float(averaged_accuracy)/(max_class-min_class), ret_str

# This accuracy is not the simple accuracy
# It count all the instances which belong to one class i
# As well as all the instances which are predicted as class i
# The the accuracy is calculated based on all the above instances
def class_based_accuracy(predict_y_vector, real_y_vector):
    min_class = min(real_y_vector)
    max_class = max(real_y_vector) + 1
    ret_list = []
    for c in range(min_class, max_class):
        real_in = np.where(real_y_vector==c)[0]
        pred_in = np.where(predict_y_vector==c)[0]
        class_index = np.unique(np.concatenate((real_in, pred_in), axis=0))
        #class_index = np.where(real_y_vector==c)
        class_predict_y = predict_y_vector[class_index]
        class_real_y = real_y_vector[class_index]
        class_accuracy = accuracy_score(class_real_y, class_predict_y, True)
        ret_list.append(class_accuracy)
    return np.array(ret_list)


def predict_matrix_with_prob_to_predict_accuracy(predict_prob_matrix, real_y_vector):
    predict_y_vector = np.argmax(predict_prob_matrix, axis=1)
    return accuracy_score(real_y_vector, predict_y_vector), predict_y_vector

# For binary class classification only
# 1 means positive class and 0 means negative class label
def f1_value_precision_recall_accuracy(predict_y_vector, real_y_vector, major_class=1):
    if len(predict_y_vector) != len(real_y_vector):
        raise Exception("Length for prediction is not same")
    #if (max(predict_y_vector) != max(real_y_vector)) or (min(predict_y_vector) != min(real_y_vector)):
    #    raise Exception("max or min prediction is not same")

    instance_num = len(predict_y_vector)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    accuracy = 0
    for i in range(0, instance_num):
        predict = int(predict_y_vector[i])
        real = int (real_y_vector[i])
        if real == major_class:
            if predict == real:
                tp = tp + 1
                accuracy = accuracy + 1
            else:
                fn = fn + 1
        else:
            if predict == real:
                tn = tn + 1
                accuracy = accuracy + 1
            else:
                fp = fp + 1

    accuracy = float(accuracy) / float(instance_num)
    if tp == 0:
        precision = 0
        recall = 0
        f1_value = 0
    else:
        precision = float(tp)/float(tp + fp)

        recall = float(tp)/float(tp + fn)

        f1_value = float(2 * precision * recall) / float(precision + recall)

    return accuracy, precision, recall, f1_value, tp, fp, tn, fn


def multiple_f1_value_precision_recall_accuracy(predict_y_vector, real_y_vector, logger=None):
    if logger == None:
        logger = init_logging('')
    if len(predict_y_vector) != len(real_y_vector):
        raise Exception("Length for prediction is not same")

    min_class = min(real_y_vector)
    max_class = max(real_y_vector)

    instance_num = len(predict_y_vector)
    f1_value_list = []
    for i in range(min_class, max_class+1):
        class_predict_y = np.where(predict_y_vector == i, 1, 0)
        class_real_y = np.where(real_y_vector ==i, 1, 0)
        #print class_predict_y
        #print class_real_y
        #print "==="
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for instance_index in range(0, instance_num):
            predict = int(class_predict_y[instance_index])
            real = int (class_real_y[instance_index])
            if real == 1:
                if predict == real:
                    tp = tp + 1
                else:
                    fn = fn + 1
            else:
                if predict == real:
                    tn = tn + 1
                else:
                    fp = fp + 1
    
        if tp == 0:
            precision = 0
            recall = 0
            f1_value = 0
        else:
            precision = float(tp)/float(tp + fp)
            recall = float(tp)/float(tp + fn)
            f1_value = float(2 * precision * recall) / float(precision + recall)
        f1_value_list.append(f1_value)
    f1_value_list = np.array(f1_value_list)

    accuracy = 0
    for instance_index in range(0, instance_num):
        predict = int(predict_y_vector[instance_index])
        real = int (real_y_vector[instance_index])
        if predict == real:
            accuracy = accuracy + 1
    accuracy = float(accuracy)/float(instance_num)
    return accuracy, f1_value_list

if __name__ == '__main__':
    real_y_vector = np.array([0,0,0,1,1,1,1,1,1,1])
    predict_y_vector = np.array([1,0,0,2,1,1,0,2,2,3])

    print(softmax(real_y_vector))
    sdfsd

    #predict_y_vector = np.array([0,0,0,0,0,0,0,0,0,0])
    #predict_y_vector = np.array([1,1,1,1,1,1,1,1,1,1])
    averaged_class_based_accuracy(predict_y_vector, real_y_vector)

    sdf
    accuracy, f1_value_list = multiple_f1_value_precision_recall_accuracy(predict_y_vector, real_y_vector)
    print(accuracy)
    print(f1_value_list)
