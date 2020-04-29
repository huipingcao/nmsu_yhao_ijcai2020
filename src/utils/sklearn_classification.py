import numpy as np
import sys
import time
import gc
from sklearn import neighbors
from sklearn import svm
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from svmutil import *

from collections import Counter
from log_io import init_logging
###



def gene_projected_lda_feature(train_x_matrix, train_y_vector):
    norm_time = 0
    start_time = time.time()
    train_norm_vector = np.linalg.norm(train_x_matrix, axis=0, ord=np.inf)[None, :]

    train_x_matrix = np.true_divide(train_x_matrix, train_norm_vector, where=(train_norm_vector!=0))
    norm_time = time.time() - start_time
    train_x_matrix[np.isnan(train_x_matrix)] = 0
    train_x_matrix[np.isinf(train_x_matrix)] = 1
    
    min_class = min(train_y_vector)
    max_class = max(train_y_vector)
    ret_feature_matrix = []
    lda_time = 0
    start_time = time.time()
    clf = LinearDiscriminantAnalysis()
    lda_time =  lda_time + time.time() - start_time
    for i in range(min_class, max_class+1):
        temp_y_vector = np.where(train_y_vector==i, 0, 1)
        #print "FIT"
        #print len(train_x_matrix)
        #print len(temp_y_vector)
        start_time = time.time()
        clf.fit(train_x_matrix, temp_y_vector)
        lda_time =  lda_time + time.time() - start_time
        ret_feature_matrix.append(clf.coef_)
    
    ret_feature_matrix = np.squeeze(np.array(ret_feature_matrix))
    ret_feature_matrix = np.absolute(ret_feature_matrix)
    #print ret_feature_matrix
    #print "Function end: gen_projected_lda_feature"
    return ret_feature_matrix, norm_time, lda_time


def bi_gene_lda_model(train_x_matrix, train_y_vector):
    clf = LinearDiscriminantAnalysis(priors=[0.5, 0.5])
    #print train_x_matrix.shape
    #print train_y_vector.shape
    start_time = time.time()
    clf.fit(train_x_matrix, train_y_vector)
    train_time = time.time() - start_time
    return clf, train_time


def gene_lda_model(train_x_matrix, train_y_vector):
    clf = LinearDiscriminantAnalysis()
    #print train_x_matrix.shape
    #print train_y_vector.shape
    start_time = time.time()
    clf.fit(train_x_matrix, train_y_vector)
    train_time = time.time() - start_time
    return clf, train_time

def run_lda(train_x_matrix, train_y_vector, test_x_matrix, test_y_vector, proba=False):
    clf, train_time = gene_lda_model(train_x_matrix, train_y_vector)
    if proba == True:
        predict_y = clf.predict(test_x_matrix)
        start_time = time.time()
        predict_y_proba = clf.predict_proba(test_x_matrix)
        test_time = time.time() - start_time
    else:
        start_time = time.time()
        predict_y = clf.predict(test_x_matrix)
        test_time = time.time() - start_time
        predict_y_proba = None

    return accuracy_score(test_y_vector, predict_y), predict_y, predict_y_proba, train_time, test_time


def run_rf(train_x_matrix, train_y_vector, test_x_matrix, test_y_vector, samples_leaf=20, proba=False):
    np.random.seed(0)
    #positive_index = np.where(train_y_vector==1)
    #negative_index = np.where(train_y_vector==0)
    #len_positive = len(np.where(train_y_vector == 1)[0])
    #len_negative = len(train_y_vector) - len_positive
    #logger.info("positive: " + str(len_positive))
    #logger.info("negative: " + str(len_negative))
    #if len_positive > len_negative:
    #    add_pare = '-w0 ' + str(len_positive/len_negative) + ' -w1 1'
    #else:
    #    add_pare = '-w1 ' + str(len_negative/len_positive) + ' -w0 1'
    clf = RandomForestClassifier(min_samples_leaf=samples_leaf, class_weight='balanced')
    start_time = time.time()
    clf.fit(train_x_matrix, train_y_vector)
    train_time = time.time() - start_time
    start_time = time.time()
    predict_y = clf.predict(test_x_matrix)
    test_time = time.time() - start_time
    if proba is False:
        predict_y_proba = None
    else:
        predict_y_proba = clf.predict_proba(test_x_matrix)
    return accuracy_score(test_y_vector, predict_y), predict_y, predict_y_proba, train_time, test_time



# train_x_matrix: row_num * col_num, train_y_vector: vector
def run_dt(train_x_matrix, train_y_vector, test_x_matrix, test_y_vector, n_neighbors, proba=False):
    clf = DecisionTreeClassifier(random_state=0, class_weight='balanced')
    #n_estimators = 10
    #clf = OneVsRestClassifier(BaggingClassifier(neighbors.KNeighborsClassifier(n_neighbors, weights="distance"), max_samples=1.0 / n_estimators, n_estimators=n_estimators))
    start_time = time.time()
    clf.fit(train_x_matrix, train_y_vector)
    train_time = time.time() - start_time
    start_time = time.time()
    predict_y = clf.predict(test_x_matrix)
    test_time = time.time() - start_time
    if proba == False:
        predict_y_proba = None
    else:
        predict_y_proba = clf.predict_proba(test_x_matrix)
    return accuracy_score(test_y_vector, predict_y), predict_y, predict_y_proba, train_time, test_time




# train_x_matrix: row_num * col_num, train_y_vector: vector
def run_knn(train_x_matrix, train_y_vector, test_x_matrix, test_y_vector, n_neighbors, proba=False):
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights="distance")
    #n_estimators = 10
    #clf = OneVsRestClassifier(BaggingClassifier(neighbors.KNeighborsClassifier(n_neighbors, weights="distance"), max_samples=1.0 / n_estimators, n_estimators=n_estimators))
    start_time = time.time()
    clf.fit(train_x_matrix, train_y_vector)
    train_time = time.time() - start_time
    start_time = time.time()
    predict_y = clf.predict(test_x_matrix)
    test_time = time.time() - start_time
    if proba == False:
        predict_y_proba = None
    else:
        predict_y_proba = clf.predict_proba(test_x_matrix)
    return accuracy_score(test_y_vector, predict_y), predict_y, predict_y_proba, train_time, test_time


# run knn method with returning distance values
# train_x_matrix: numpy matrix with N * A: N: is number of training instances, A is number of attributes
# train_y_vector: numpy vector N * 1
# test_x_matrix: numpy matrix with N1 * A: N1 is the number of testing instances
# test_y_vector: numpy vector N1 * 1
# n_neighbors: top K value
# it returns three values
# distances: a numpy matrix D with N1 * n_neighbors, D_ij means the distance from test instance i to the jth nearest training instance
# indexes: a numpy matrix I with N1 * n_neighbors, it records the corresponding index for the jth nearest training instance
# the distance calculation: from [A11, A12, A13] to [A21, A22, A23] is dist = sqrt((A11-A21)^2 + (A12-A22)^2 + (A13-A23)^2)
def run_knn_with_dist(train_x_matrix, train_y_vector, test_x_matrix, test_y_vector, n_neighbors=1):
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights="distance")
    min_class = min(train_y_vector)
    max_class = max(train_y_vector)
    distance_matrix = []
    for i in range(min_class, max_class+1):
        train_index = np.where(train_y_vector==i)[0]
        knn_model = clf.fit(train_x_matrix[train_index, :], train_y_vector[train_index])
        distances, indexes = knn_model.kneighbors(test_x_matrix, n_neighbors, True)
        distance_matrix.append(distances)


    distance_matrix = np.array(distance_matrix).reshape(max_class-min_class+1, len(test_y_vector))
    distance_matrix = distance_matrix.T

    start_time = time.time()
    knn_model = clf.fit(train_x_matrix, train_y_vector)
    train_time = time.time() - start_time
    start_time = time.time()
    predict_y_vector = knn_model.predict(test_x_matrix)
    test_time = time.time() - start_time
    return distance_matrix, predict_y_vector, train_time, test_time


def get_pred_matrix(train_y_vector, index_matrix):
    x_row, x_col = index_matrix.shape
    pred_matrix = np.zeros([x_row, x_col]).astype(int)
    for i in range(0, x_row):
        pred_matrix[i] = train_y_vector[index_matrix[i]]

    return pred_matrix


# find the first instance belongs to current_class d1 and the first instance does not belong to the current_class d2
# then the probability belongs to current_class is 1- (d1/(d1 + d2)) and the probability it does not belong to current_class is 1 - (d2/(d1 + d2))
# This function will transfer all class labels to be one continues vector wihch starts from 0
# return test_x_matrix_row * num_classes matrix, it contains the probability dist for each class
# the small return value means the higher probability
def run_knn_with_proba(train_x_matrix, train_y_vector, test_x_matrix, test_y_vector):
    train_row, train_col = train_x_matrix.shape
    test_row, test_col = test_x_matrix.shape
    min_class = min(train_y_vector)
    if min_class != 0:
        train_y_vector = train_y_vector - min_class
    min_class = 0
    max_class = max(train_y_vector)
    num_classes = max_class + 1

    dist_matrix, index_matrix, knn_model, train_time, test_time = run_knn_with_dist(train_x_matrix, train_y_vector, test_x_matrix, test_y_vector, n_neighbors)
    start_time = time.time()
    pred_matrix = get_pred_matrix(train_y_vector, index_matrix)

    max_dist = dist_matrix.max() + 1.0
    predict_proba_matrix = np.full([test_row, num_classes], max_dist)
    predict_proba_matrix = knn_model.predict_proba(test_x_matrix)
    #for i in range(0, test_row):
    #    instance_pred_vector = pred_matrix[i]
    #    pred_len = len(instance_pred_vector)
    #    for j in range(0, pred_len):
    #        c = instance_pred_vector[j]
    #        if predict_proba_matrix[i][c] != max_dist:
    #            continue
    #        predict_proba_matrix[i][c] = dist_matrix[i][j]

    #predict_proba_matrix = predict_proba_matrix

    #test_time = test_time + time.time() - start_time
    #for i in range(0, test_row):
    #    proba_vector = predict_proba_matrix[i]
    #    vector_min = proba_vector.min()
    #    predict_proba_matrix[i] = 1- (predict_proba_matrix[i] - vector_min)/(max_dist - vector_min)
    #predict_proba_matrix = (predict_proba_matrix - predict_proba_matrix.min(axis=0))/ (predict_proba_matrix.max(axis=0) - predict_proba_matrix.min(axis=0))
    #print predict_proba_matrix


    #for i in range(0, test_row):
    #    proba_vector = predict_proba_matrix[i]
    #    null_index = np.where(proba_vector==-1)
    #    not_null_index = np.where(proba_vector!=-1)[0]
    #    if len(not_null_index) == 1:
    #        predict_proba_matrix[i][not_null_index] = 1
    #    else:
    #        proba_vector = np.delete(proba_vector, null_index)
    #        sum_proba = sum(proba_vector)
    #        for j in not_null_index:
    #            predict_proba_matrix[i][j] = predict_proba_matrix[i][j]/sum_proba
    #    predict_proba_matrix[i][null_index] = 0

    return predict_proba_matrix, train_time, test_time


# Libsvm
def run_sklearn_libsvm(train_x_matrix, train_y_vector, test_x_matrix, test_y_vector, proba=False):
    train_y_vector = train_y_vector- min(train_y_vector)
    test_y_vector = test_y_vector - min(test_y_vector)
    train_x_matrix = train_x_matrix.astype(np.float64)
    train_y_vector = train_y_vector.astype(np.float64)
    test_x_matrix = test_x_matrix.astype(np.float64)
    test_y_vector = test_y_vector.astype(np.float64)

    weight_array = []
    unique, counts = np.unique(train_y_vector, return_counts=True)
    
    count_all = len(train_y_vector)
    for i in counts:
        weight_array.append(float(1)/i)
    weight_array = np.array(weight_array)

    start_time = time.time()
    model = svm.libsvm.fit(train_x_matrix, train_y_vector, class_weight=weight_array)
    train_time = time.time() - start_time
    start_time = time.time()
    predict_y = svm.libsvm.predict(test_x_matrix, *model)
    test_time = time.time() - start_time
    if proba is False:
        predict_y_proba = None
    else:
        predict_y_proba = svm.libsvm.predict_proba(test_x_matrix, *model)
        #predict_y_proba = None
    return accuracy_score(test_y_vector, predict_y), predict_y, predict_y_proba, train_time, test_time
    #return accuracy_score(test_y_vector, predict_y), predict_y, train_time, test_time


def banlanced_binary_processing(train_x_matrix, train_y_vector, banlanced_ratio=3):
    positive_index = np.where(train_y_vector==0.0)[0]
    negative_index = np.where(train_y_vector==1.0)[0]

    positive_len = len(positive_index)
    negative_len = len(negative_index)
    if positive_len > negative_len:
        select_len = banlanced_ratio * negative_len
        if positive_len > select_len:
            select_index = np.random.choice(positive_len, select_len, replace=False)
            positive_index = positive_index[select_index]
            all_index = np.append(positive_index, negative_index)
            train_x_matrix = train_x_matrix[all_index, :]
            train_y_vector = train_y_vector[all_index]
    else:
        select_len = banlanced_ratio * positive_len
        if negative_len > select_len:
            select_index = np.random.choice(negative_len, select_len, replace=False)
            negative_index = negative_index[select_index]
            all_index = np.append(negative_index, positive_index)
            train_x_matrix = train_x_matrix[all_index, :]
            train_y_vector = train_y_vector[all_index]

    return train_x_matrix, train_y_vector



def libsvm_load_predict(test_x_matrix, test_y_vector, save_file):
    model = svm_load_model(save_file)
    predict_y, predict_acc, predict_y_proba = svm_predict(test_y_vector, test_x_matrix, model, '-b 1')
    print(predict_acc, predict_y, predict_y_proba)

#libsvm from the author's website
def run_libsvm(train_x_matrix, train_y_vector, test_x_matrix, test_y_vector, logger, proba=False, save_file='', weight=True):
    train_y_vector = train_y_vector- min(train_y_vector)
    test_y_vector = test_y_vector - min(test_y_vector)
    #train_x_matrix = train_x_matrix.astype(np.float64)
    #train_y_vector = train_y_vector.astype(np.float64)
    #test_x_matrix = test_x_matrix.astype(np.float64)
    #test_y_vector = test_y_vector.astype(np.float64)

    if weight == True:
        positive_index = np.where(train_y_vector==1)
        negative_index = np.where(train_y_vector==0)
        len_positive = len(np.where(train_y_vector == 1)[0])
        len_negative = len(train_y_vector) - len_positive

        logger.info("positive: " + str(len_positive))
        logger.info("negative: " + str(len_negative))

        if len_positive > len_negative:
            add_pare = '-w0 ' + str(len_positive/len_negative) + ' -w1 1'
        else:
            add_pare = '-w1 ' + str(len_negative/len_positive) + ' -w0 1'
    else:
        add_pare = ''

    train_x_matrix = train_x_matrix.tolist()
    train_y_vector = train_y_vector.astype(np.integer).tolist()
    test_x_matrix = test_x_matrix.tolist()
    test_y_vector = test_y_vector.astype(np.integer).tolist()

    #svm_model.predict = lambda self, x: svm_predict([0], [x], self)[0][0]
    #prob = svm_problem([1,-1], [[1,0,1], [-1,0,-1]])
    prob = svm_problem(train_y_vector, train_x_matrix)
    
    #logger.info("libsvm parameter: " + '-h 0 -s 0 -t 2 -c 0.03125 -g 0.0078125 -b 1 '+add_pare)
    #param = svm_parameter('-h 0 -s 0 -t 2 -c 0.03125 -g 0.0078125 -b 1 '+add_pare)
    logger.info("libsvm parameter: " + '-h 0 -s 0 -t 2 -b 1 -e 0.1 '+add_pare)
    param = svm_parameter('-h 0 -s 0 -t 2 -b 1 -e 0.1 '+add_pare)

    start_time = time.time()
    model = svm_train(prob, param)
    train_time = time.time() - start_time

    if save_file != '':
        logger.info("svm model saved to " + save_file)
        svm_save_model(save_file, model)

    start_time = time.time()
    #predict_y, predict_acc, predict_val = svm_predict(test_y_vector, test_x_matrix, model, '-b 1')
    predict_y, predict_acc, predict_val = svm_predict(test_y_vector, test_x_matrix, model)
    test_time = time.time() - start_time
    #predict_val = np.array(predict_val)
    #predict_y = np.array(predict_y)
    #print predict_val.shape
    #print predict_y.shape
    predict_y = np.array(predict_y)
    predict_val = np.zeros([len(predict_y), 2])
    predict_val[:, 0] = 1 - predict_y
    predict_val[:, 1] = predict_y
    return predict_acc[0], predict_y, predict_val, train_time, test_time


def run_svm_svc(train_x_matrix, train_y_vector, test_x_matrix, test_y_vector, proba=False):
    clf = svm.SVC()
    print("kernel function:")
    print(clf.kernel)
    print(clf.decision_function_shape)
    print(clf.degree)
    print(clf.C)
    clf.probability = proba
    start_time = time.time()
    clf.fit(train_x_matrix, train_y_vector)
    train_time = time.time() - start_time
    start_time = time.time()
    predict_y = clf.predict(test_x_matrix)
    test_time = time.time() - start_time
    if proba == False:
        predict_y_proba = None
    else:
        predict_y_proba = clf.predict_proba(test_x_matrix)
    return accuracy_score(test_y_vector, predict_y), predict_y, predict_y_proba, train_time, test_time


def run_nn(train_x_matrix, train_y_vector, test_x_matrix, test_y_vector, proba=False):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    start_time = time.time()
    clf.fit(train_x_matrix, train_y_vector)
    train_time = time.time() - start_time
    start_time = time.time()
    predict_y = clf.predict(test_x_matrix)
    test_time = time.time() - start_time
    if proba == False:
        predict_y_proba = None
    else:
        predict_y_proba = clf.predict_proba(test_x_matrix)
    return accuracy_score(test_y_vector, predict_y), predict_y, predict_y_proba, train_time, test_time




##############################
# we only consider KNN with K=1
def run_feature_knn_use_proba(train_x_matrix, train_y_vector, test_x_matrix, test_y_vector, feature_array, attr_num, n_neighbors, class_id=-1, logger=None):
    if logger==None:
        logger = init_logging("")
        logger.info('no log file: ')

    num_classes, num_features = feature_array.shape
    test_row, test_col = test_x_matrix.shape
    knn_predict_matrix = np.zeros(test_row * num_classes).reshape(test_row, num_classes)
    knn_predict_proba = np.zeros(test_row * num_classes).reshape(test_row, num_classes)

    knn_train_time = 0
    knn_test_time = 0
    
    knn_accuracy = 0
    proba = True 
    if class_id == -1:
        min_class = min(train_y_vector)
        max_class = max(train_y_vector) + 1
    else:
        min_class = class_id
        max_class = class_id + 1
    #result_matrix = np.zeros((10, num_classes))
    for i in range(min_class, max_class):
        logger.info('class: ' +str(i))
        logger.info(str(feature_array[i]))
        #print 'class: ' + str(i)
        #print feature_array[i]
        temp_train_x_matrix, temp_attr_num, temp_attr_len = feature_data_generation(train_x_matrix, attr_num, feature_array[i])
        temp_test_x_matrix, temp_attr_num, temp_attr_len = feature_data_generation(test_x_matrix, attr_num, feature_array[i])
        #print 'class: ' + str(i)
        temp_train_y_vector = np.where(train_y_vector==i, 1, 0)
        temp_test_y_vector = np.where(test_y_vector==i, 1, 0)
        if i==0:
            logger.info('sub feature data shape: ')
            logger.info(str(temp_train_x_matrix.shape))
            logger.info(str(temp_test_x_matrix.shape))
            #print 'sub feature data shape:'
            #print temp_train_x_matrix.shape
            #print temp_test_x_matrix.shape
        
        temp_accuracy, temp_predict_y, temp_predict_y_proba, temp_train_time, temp_test_time = run_knn(temp_train_x_matrix, temp_train_y_vector, temp_test_x_matrix, temp_test_y_vector, n_neighbors, proba)
        
        #temp_predict_y_proba, temp_predict_y, temp_train_time, temp_test_time = run_knn_with_dist(temp_train_x_matrix, temp_train_y_vector, temp_test_x_matrix, temp_test_y_vector)
        
        #temp_accuracy_1, temp_precision, temp_recall, temp_f1_value = f1_value_precision_recall_accuracy(temp_predict_y, temp_test_y_vector)
        temp_accuracy, temp_precision, temp_recall, temp_f1_value, temp_tp, temp_fp, temp_tn, temp_fn = f1_value_precision_recall_accuracy(temp_predict_y, temp_test_y_vector)
        #if temp_accuracy != temp_accuracy_1:
        #    logger.info(str(temp_accuracy))
        #    logger.info(str(temp_accuracy_1))
        #    #print temp_accuracy
        #    #print temp_accuracy_1
        #    raise Exception("Two accuracy results are not the same")
        #result_matrix[0, i] = temp_accuracy
        #result_matrix[1, i] = temp_precision
        #result_matrix[2, i] = temp_recall
        #result_matrix[3, i] = temp_f1_value
        #result_matrix[4, i] = temp_tp
        #result_matrix[5, i] = temp_fp
        #result_matrix[6, i] = temp_tn
        #result_matrix[7, i] = temp_fn
        #result_matrix[8, i] = temp_train_time
        #result_matrix[9, i] = temp_test_time
        logger.info("Accuracy for class " + str(i) + ": " + str(temp_accuracy))
        logger.info("Recall for class " + str(i) + ": " + str(temp_recall))
        logger.info("Precision for class " + str(i) + ": " + str(temp_precision))
        logger.info("F1 Score for class " + str(i) + ": " + str(temp_f1_value))
        logger.info("Prediction matrix:")
        logger.info("TP=" + str(temp_tp) + " FP=" + str(temp_fp))
        logger.info("TN=" + str(temp_tn) + " FN=" + str(temp_fn))


        knn_train_time = knn_train_time + temp_train_time
        knn_test_time = knn_test_time + temp_test_time

        proba_row, proba_col = temp_predict_y_proba.shape

        knn_predict_matrix[:, i] = temp_predict_y_proba[:, 1]
        logger.info('=============')
        #break

    knn_accuracy, knn_predict_y = predict_matrix_with_proba_to_predict_accuracy(knn_predict_matrix, knn_predict_matrix, test_y_vector)
    
    return knn_accuracy, knn_train_time, knn_test_time, knn_predict_y

def load_predict_svm_proba(test_x_matrix, test_y_vector, feature_array, attr_num, save_pre, logger=None):
    if logger==None:
        logger = init_logging("")
        logger.info('no log file: ')

    num_classes, num_features = feature_array.shape
    test_row, test_col = test_x_matrix.shape
    svm_predict_matrix = np.zeros(test_row * num_classes).reshape(test_row, num_classes)
    svm_predict_proba = np.zeros(test_row * num_classes).reshape(test_row, num_classes)

    svm_train_time = 0
    svm_test_time = 0
    
    svm_accuracy = 0
    proba = True 

    banlanced_ratio = 5

    for i in range(0, num_classes):
        #print 'class: ' + str(i)
        #print feature_array[i]
        logger.info("class: " + str(i))
        logger.info(str(feature_array[i]))
        
        temp_test_x_matrix, temp_attr_num, temp_attr_len = feature_data_generation(test_x_matrix, attr_num, feature_array[i])
        #print 'class: ' + str(i)
        if i==0:
            logger.info('sub feature data shape: ')
            logger.info(str(temp_train_x_matrix.shape))
            logger.info(str(temp_test_x_matrix.shape))
            
        temp_test_y_vector = np.where(test_y_vector==i, 1, 0)

        save_file = save_pre + "_class" + str(i) + ".model"

        

        temp_accuracy, temp_precision, temp_recall, temp_f1_value, temp_tp, temp_fp, temp_tn, temp_fn = f1_value_precision_recall_accuracy(temp_predict_y, temp_test_y_vector)
        temp_predict_y = np.array(temp_predict_y)
        temp_predict_y_proba = np.array(temp_predict_y_proba)

        logger.info("Accuracy for class " + str(i) + ": " + str(temp_accuracy))
        logger.info("Recall for class " + str(i) + ": " + str(temp_recall))
        logger.info("Precision for class " + str(i) + ": " + str(temp_precision))
        logger.info("F1 Score for class " + str(i) + ": " + str(temp_f1_value))
        logger.info("Prediction matrix:")
        logger.info("TP=" + str(temp_tp) + " FP=" + str(temp_fp))
        logger.info("TN=" + str(temp_tn) + " FN=" + str(temp_fn))

        svm_train_time = svm_train_time + temp_train_time
        svm_test_time = svm_test_time + temp_test_time

        proba_row, proba_col = temp_predict_y_proba.shape

        svm_predict_matrix[:, i] = temp_predict_y
        #svm_predict_proba[:, i] = temp_predict_y_proba[:, 1]
        logger.info('=============')
        #break

    svm_accuracy, svm_predict_y = predict_matrix_with_proba_to_predict_accuracy(svm_predict_matrix, svm_predict_proba, test_y_vector)
    return svm_accuracy, svm_train_time, svm_test_time, svm_predict_y


def run_feature_svm_use_proba(train_x_matrix, train_y_vector, test_x_matrix, test_y_vector, feature_array, attr_num, logger=None, save_pre=''):
    if logger==None:
        logger = init_logging("")
        logger.info('no log file: ')
    num_classes, num_features = feature_array.shape
    test_row, test_col = test_x_matrix.shape
    svm_predict_matrix = np.zeros(test_row * num_classes).reshape(test_row, num_classes)
    svm_predict_proba = np.zeros(test_row * num_classes).reshape(test_row, num_classes)

    svm_train_time = 0
    svm_test_time = 0
    
    svm_accuracy = 0
    proba = True 

    banlanced_ratio = 5

    for i in range(0, num_classes):
        #print 'class: ' + str(i)
        #print feature_array[i]
        logger.info("class: " + str(i))
        logger.info(str(feature_array[i]))
        temp_train_x_matrix, temp_attr_num, temp_attr_len = feature_data_generation(train_x_matrix, attr_num, feature_array[i])
        temp_test_x_matrix, temp_attr_num, temp_attr_len = feature_data_generation(test_x_matrix, attr_num, feature_array[i])
        #print 'class: ' + str(i)
        if i==0:
            logger.info('sub feature data shape: ')
            logger.info(str(temp_train_x_matrix.shape))
            logger.info(str(temp_test_x_matrix.shape))
            
        temp_train_y_vector = np.where(train_y_vector==i, 1, 0)
        temp_test_y_vector = np.where(test_y_vector==i, 1, 0)

        temp_train_x_matrix, temp_train_y_vector = banlanced_binary_processing(temp_train_x_matrix, temp_train_y_vector, banlanced_ratio)

        save_file = save_pre + "_class" + str(i) + "_top" + str(temp_attr_num) + ".model"

        logger.info('svm saved to ' + save_file)
        temp_accuracy, temp_predict_y, temp_predict_y_proba, temp_train_time, temp_test_time = run_libsvm(temp_train_x_matrix, temp_train_y_vector, temp_test_x_matrix, temp_test_y_vector, logger, proba, save_file)
        temp_accuracy, temp_precision, temp_recall, temp_f1_value, temp_tp, temp_fp, temp_tn, temp_fn = f1_value_precision_recall_accuracy(temp_predict_y, temp_test_y_vector)
        temp_predict_y = np.array(temp_predict_y)
        temp_predict_y_proba = np.array(temp_predict_y_proba)

        logger.info("Accuracy for class " + str(i) + ": " + str(temp_accuracy))
        logger.info("Recall for class " + str(i) + ": " + str(temp_recall))
        logger.info("Precision for class " + str(i) + ": " + str(temp_precision))
        logger.info("F1 Score for class " + str(i) + ": " + str(temp_f1_value))
        logger.info("Prediction matrix:")
        logger.info("TP=" + str(temp_tp) + " FP=" + str(temp_fp))
        logger.info("TN=" + str(temp_tn) + " FN=" + str(temp_fn))

        svm_train_time = svm_train_time + temp_train_time
        svm_test_time = svm_test_time + temp_test_time

        proba_row, proba_col = temp_predict_y_proba.shape

        svm_predict_matrix[:, i] = temp_predict_y
        svm_predict_proba[:, i] = temp_predict_y_proba[:, 1]
        logger.info('=============')
        #break

    svm_accuracy, svm_predict_y = predict_matrix_with_proba_to_predict_accuracy(svm_predict_matrix, svm_predict_proba, test_y_vector)
    return svm_accuracy, svm_train_time, svm_test_time, svm_predict_y


def run_feature_svm_load_proba(model_pre, test_x_matrix, test_y_vector, feature_array, attr_num, logger=None):
    if logger==None:
        logger = init_logging("")
        logger.info('no log file: ')
    num_classes, num_features = feature_array.shape
    test_row, test_col = test_x_matrix.shape
    svm_predict_matrix = np.zeros(test_row * num_classes).reshape(test_row, num_classes)
    svm_predict_proba = np.zeros(test_row * num_classes).reshape(test_row, num_classes)

    svm_train_time = 0
    svm_test_time = 0
    
    svm_accuracy = 0
    proba = True 
    for i in range(0, num_classes):
        #print 'class: ' + str(i)
        #print feature_array[i]
        logger.info("class: " + str(i))
        logger.info(str(feature_array[i]))
        #temp_train_x_matrix, temp_attr_num, temp_attr_len = feature_data_generation(train_x_matrix, attr_num, feature_array[i])

        temp_test_x_matrix, temp_attr_num, temp_attr_len = feature_data_generation(test_x_matrix, attr_num, feature_array[i])
        model_file = model_pre + '_class' + str(i) + "_top" + str(temp_attr_len) + ".model"
        print(model_file)
        logger.info('model file: ' + model_file)
        start_time = time.time()
        svm_model = svm_load_model(model_file)
        temp_train_time = time.time() - start_time
        svm_train_time = svm_train_time + temp_train_time

        #print 'class: ' + str(i)
        if i==0:
            logger.info('sub feature data shape: ')
            logger.info(str(temp_test_x_matrix.shape))
            
        temp_test_y_vector = np.where(test_y_vector==i, 1, 0)
        temp_test_x_matrix = temp_test_x_matrix.tolist()
        temp_test_y_vector = temp_test_y_vector.astype(np.integer).tolist()

        ###START FROM HERE
        start_time = time.time()
        temp_predict_y, temp_accuracy, temp_predict_y_proba = svm_predict(temp_test_y_vector, temp_test_x_matrix, svm_model, '-b 1')
        temp_test_time = time.time() - start_time
        svm_train_time = svm_train_time + temp_test_time

        temp_accuracy, temp_precision, temp_recall, temp_f1_value, temp_tp, temp_fp, temp_tn, temp_fn = f1_value_precision_recall_accuracy(temp_predict_y, temp_test_y_vector)
        temp_predict_y = np.array(temp_predict_y)
        temp_predict_y_proba = np.array(temp_predict_y_proba)

        logger.info("Accuracy for class " + str(i) + ": " + str(temp_accuracy))
        logger.info("Recall for class " + str(i) + ": " + str(temp_recall))
        logger.info("Precision for class " + str(i) + ": " + str(temp_precision))
        logger.info("F1 Score for class " + str(i) + ": " + str(temp_f1_value))
        logger.info("Prediction matrix:")
        logger.info("TP=" + str(temp_tp) + " FP=" + str(temp_fp))
        logger.info("TN=" + str(temp_tn) + " FN=" + str(temp_fn))

        proba_row, proba_col = temp_predict_y_proba.shape

        svm_predict_matrix[:, i] = temp_predict_y
        svm_predict_proba[:, i] = temp_predict_y_proba[:, 1]
        logger.info('=============')
        #break
    svm_accuracy, svm_predict_y = predict_matrix_with_proba_to_predict_accuracy(svm_predict_matrix, svm_predict_proba, test_y_vector)
    return svm_accuracy, svm_train_time, svm_test_time, svm_predict_y


def run_feature_nn_use_proba(train_x_matrix, train_y_vector, test_x_matrix, test_y_vector, feature_array, attr_num, start_class=0):
    num_classes, num_features = feature_array.shape
    test_row, test_col = test_x_matrix.shape

    nn_predict_matrix = np.zeros(test_row * num_classes).reshape(test_row, num_classes)
    nn_predict_proba = np.zeros(test_row * num_classes).reshape(test_row, num_classes)

    nn_train_time = 0
    nn_test_time = 0
    
    nn_accuracy = 0
    proba = True 
    
    for i in range(0, num_classes):
        print('class: ' + str(i))
        print(feature_array[i])
        temp_train_x_matrix, temp_attr_num, temp_attr_len = feature_data_generation(train_x_matrix, attr_num, feature_array[i])
        temp_test_x_matrix, temp_attr_num, temp_attr_len = feature_data_generation(test_x_matrix, attr_num, feature_array[i])
        #print 'class: ' + str(i)
        if i==0:
            print('sub feature data shape:')
            print(temp_train_x_matrix.shape)
            print(temp_test_x_matrix.shape)
        temp_accuracy, temp_predict_y, temp_predict_y_proba, temp_train_time, temp_test_time = run_nn(temp_train_x_matrix, train_y_vector, temp_test_x_matrix, test_y_vector, proba)
        nn_train_time = nn_train_time + temp_train_time
        nn_test_time = nn_test_time + temp_test_time
        #nn_predict_proba = np.add(nn_predict_proba, temp_predict_y_proba)
        #print nn_predict_proba.shape
        #print temp_predict_y_proba.shape
        nn_predict_matrix[:, i] = temp_predict_y
        nn_predict_proba[:, i] = temp_predict_y_proba[:, i]
        #break

    nn_accuracy, nn_predict_y = predict_matrix_with_proba_to_predict_accuracy(nn_predict_matrix, nn_predict_proba, test_y_vector)
    return nn_accuracy, nn_train_time, nn_test_time, nn_predict_y


def run_feature_lda_use_proba(train_x_matrix, train_y_vector, test_x_matrix, test_y_vector, feature_array, attr_num, logger=None):
    if logger==None:
        logger = init_logging("")
        logger.info('no log file: ')
    num_classes, num_features = feature_array.shape
    test_row, test_col = test_x_matrix.shape
    lda_predict_matrix = np.zeros(test_row * num_classes).reshape(test_row, num_classes)
    lda_predict_proba = np.zeros(test_row * num_classes).reshape(test_row, num_classes)

    lda_train_time = 0
    lda_test_time = 0
    
    lda_accuracy = 0
    proba = True 
    for i in range(0, num_classes):
        logger.info("class: " + str(i))
        logger.info(str(feature_array[i]))
        temp_train_x_matrix, temp_attr_num, temp_attr_len = feature_data_generation(train_x_matrix, attr_num, feature_array[i])
        temp_test_x_matrix, temp_attr_num, temp_attr_len = feature_data_generation(test_x_matrix, attr_num, feature_array[i])
        
        if i==0:
            logger.info('sub feature data shape: ')
            logger.info(str(temp_train_x_matrix.shape))
            logger.info(str(temp_test_x_matrix.shape))
            
        temp_train_y_vector = np.where(train_y_vector==i, 1, 0)
        temp_test_y_vector = np.where(test_y_vector==i, 1, 0)

        temp_accuracy, temp_predict_y, temp_predict_y_proba, temp_train_time, temp_test_time = run_lda(temp_train_x_matrix, temp_train_y_vector, temp_test_x_matrix, temp_test_y_vector, proba)
        temp_accuracy, temp_precision, temp_recall, temp_f1_value, temp_tp, temp_fp, temp_tn, temp_fn = f1_value_precision_recall_accuracy(temp_predict_y, temp_test_y_vector)

        logger.info("Accuracy for class " + str(i) + ": " + str(temp_accuracy))
        logger.info("Recall for class " + str(i) + ": " + str(temp_recall))
        logger.info("Precision for class " + str(i) + ": " + str(temp_precision))
        logger.info("F1 Score for class " + str(i) + ": " + str(temp_f1_value))
        logger.info("Prediction matrix:")
        logger.info("TP=" + str(temp_tp) + " FP=" + str(temp_fp))
        logger.info("TN=" + str(temp_tn) + " FN=" + str(temp_fn))

        lda_train_time = lda_train_time + temp_train_time
        lda_test_time = lda_test_time + temp_test_time

        proba_row, proba_col = temp_predict_y_proba.shape

        lda_predict_matrix[:, i] = temp_predict_y
        lda_predict_proba[:, i] = temp_predict_y_proba[:, 1]
        logger.info('=============')
        #break

    lda_accuracy, lda_predict_y = predict_matrix_with_proba_to_predict_accuracy(lda_predict_matrix, lda_predict_proba, test_y_vector)
    return lda_accuracy, lda_train_time, lda_test_time, lda_predict_y



def run_feature_rf_use_proba(train_x_matrix, train_y_vector, test_x_matrix, test_y_vector, feature_array, attr_num, logger=None):
    if logger==None:
        logger = init_logging("")
        logger.info('no log file: ')
    num_classes, num_features = feature_array.shape
    test_row, test_col = test_x_matrix.shape
    rf_predict_matrix = np.zeros(test_row * num_classes).reshape(test_row, num_classes)
    rf_predict_proba = np.zeros(test_row * num_classes).reshape(test_row, num_classes)

    rf_train_time = 0
    rf_test_time = 0
    
    rf_accuracy = 0
    proba = True 
    for i in range(0, num_classes):
        logger.info("class: " + str(i))
        logger.info(str(feature_array[i]))
        temp_train_x_matrix, temp_attr_num, temp_attr_len = feature_data_generation(train_x_matrix, attr_num, feature_array[i])
        temp_test_x_matrix, temp_attr_num, temp_attr_len = feature_data_generation(test_x_matrix, attr_num, feature_array[i])
        
        if i==0:
            logger.info('sub feature data shape: ')
            logger.info(str(temp_train_x_matrix.shape))
            logger.info(str(temp_test_x_matrix.shape))
            
        temp_train_y_vector = np.where(train_y_vector==i, 1, 0)
        temp_test_y_vector = np.where(test_y_vector==i, 1, 0)

        temp_accuracy, temp_predict_y, temp_predict_y_proba, temp_train_time, temp_test_time = run_rf(temp_train_x_matrix, temp_train_y_vector, temp_test_x_matrix, temp_test_y_vector, 20, True)
        temp_accuracy, temp_precision, temp_recall, temp_f1_value, temp_tp, temp_fp, temp_tn, temp_fn = f1_value_precision_recall_accuracy(temp_predict_y, temp_test_y_vector)

        logger.info("Accuracy for class " + str(i) + ": " + str(temp_accuracy))
        logger.info("Recall for class " + str(i) + ": " + str(temp_recall))
        logger.info("Precision for class " + str(i) + ": " + str(temp_precision))
        logger.info("F1 Score for class " + str(i) + ": " + str(temp_f1_value))
        logger.info("Prediction matrix:")
        logger.info("TP=" + str(temp_tp) + " FP=" + str(temp_fp))
        logger.info("TN=" + str(temp_tn) + " FN=" + str(temp_fn))

        rf_train_time = rf_train_time + temp_train_time
        rf_test_time = rf_test_time + temp_test_time

        proba_row, proba_col = temp_predict_y_proba.shape

        rf_predict_matrix[:, i] = temp_predict_y
        rf_predict_proba[:, i] = temp_predict_y_proba[:, 1]
        logger.info('=============')
        #break

    rf_accuracy, rf_predict_y = predict_matrix_with_proba_to_predict_accuracy(rf_predict_matrix, rf_predict_proba, test_y_vector)
    return rf_accuracy, rf_train_time, rf_test_time, rf_predict_y



def run_feature_rf_use_proba_old(train_x_matrix, train_y_vector, test_x_matrix, test_y_vector, feature_array, attr_num, logger=None):
    num_classes, num_features = feature_array.shape
    test_row, test_col = test_x_matrix.shape
    rf_predict_proba = np.zeros(test_row * num_classes).reshape(test_row, num_classes)

    rf_train_time = 0
    rf_test_time = 0
    
    rf_accuracy = 0
    
    for i in range(0, num_classes):
        print('class: ' + str(i))
        temp_train_x_matrix, temp_attr_num, temp_attr_len = feature_data_generation(train_x_matrix, attr_num, feature_array[i])
        temp_test_x_matrix, temp_attr_num, temp_attr_len = feature_data_generation(test_x_matrix, attr_num, feature_array[i])
        #print 'class: ' + str(i)
        if i==0:
            print('sub feature data shape:')
            print(temp_train_x_matrix.shape)
            print(temp_test_x_matrix.shape)
        temp_accuracy, temp_predict_y, temp_predict_y_proba, temp_train_time, temp_test_time = run_rf(temp_train_x_matrix, train_y_vector, temp_test_x_matrix, test_y_vector)
        rf_train_time = rf_train_time + temp_train_time
        rf_test_time = rf_test_time + temp_test_time
        if temp_predict_y_proba != None:
            temp_predict_y_proba[:, i] = temp_predict_y_proba[:, i]
            print(temp_predict_y_proba)
        rf_predict_proba = np.add(rf_predict_proba, temp_predict_y_proba)
        #rf_predict_proba[:, i] = temp_predict_y_proba[:, i]
        #break

    rf_accuracy, rf_predict_y = predict_proba_to_predict_accuracy(rf_predict_proba, test_y_vector, start_class)
    return rf_accuracy, rf_train_time, rf_test_time, rf_predict_y






if __name__ == '__main__':
    train_x_matrix = np.array([[-1, -4, -7], [2, -1, 7], [-3, 2, 7], [1, 1, 7], [2, 1, 7], [3, 2, 7]]).astype(np.float64)
    test_x_matrix = np.array([[12, 5, 7], [2, 1, -7], [-3, -2, -7], [-1, 1, -7]]).astype(np.float64)

    train_y_vector = np.array([0, 1, 1, 1, 1, 1]).astype(np.float64)
    test_y_vector = np.array([1, 0, 0, 0]).astype(np.float64)
