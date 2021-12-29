# SVMs for Food Experiment Data
import matplotlib 
import numpy as np
import matplotlib.pyplot as pp
import optparse
import unittest
import random
import itertools
from sklearn import decomposition
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.ndimage.filters import gaussian_filter1d
import sys
import lib_get_subset as lib_get_subset
import lib_overlapping as lib_overlapping
import matplotlib.pyplot as plt

try:
    import cPickle as pkl
    def load_pickle(filename):
        with open(filename, 'rb') as f:
            return pkl.load(f)
except:
    import pickle as pkl
    def load_pickle(filename):
        with open(filename, 'rb') as f:
            return pkl.load(f, encoding = 'latin1')


TRIAL_ORDER = [[1, 4, 6, 2, 3, 7, 0, 5, 8],
               [0, 3, 6, 2, 5, 7, 1, 4, 8],
               [2, 3, 7, 1, 4, 6, 0, 5, 8],
               [2, 4, 6, 0, 5, 8, 1, 3, 7],
               [0, 5, 6, 1, 4, 7, 2, 3, 8],
               [1, 3, 8, 0, 4, 6, 2, 5, 7],
               [0, 4, 7, 1, 3, 8, 2, 5, 6],
               [2, 3, 8, 0, 4, 6, 1, 5, 7],
               [0, 3, 8, 2, 4, 6, 1, 5, 7],
               [1, 3, 6, 0, 5, 7, 2, 4, 8],
               [0, 5, 7, 1, 3, 8, 2, 4, 6],
               [1, 3, 8, 0, 4, 7, 2, 5, 6],
               [1, 5, 6, 0, 4, 7, 2, 3, 8],
               [1, 5, 8, 2, 4, 6, 0, 3, 7],
               [1, 5, 8, 2, 4, 6, 0, 3, 7],
               [1, 4, 6, 2, 5, 7, 0, 3, 8],
               [0, 4, 6, 2, 5, 8, 1, 3, 7],
               [2, 4, 7, 0, 5, 6, 1, 3, 8],
               [1, 5, 7, 0, 3, 6, 2, 4, 8],
               [2, 5, 7, 0, 4, 8, 1, 3, 6],
               [2, 5, 7, 1, 4, 6, 0, 3, 8],
               [0, 5, 7, 2, 3, 6, 1, 4, 8],
               [2, 4, 8, 0, 5, 6, 1, 3, 7],
               [2, 5, 7, 0, 4, 8, 1, 3, 6],
               [1, 3, 7, 2, 5, 8, 0, 4, 6],
               [2, 3, 8, 1, 5, 7, 0, 4, 6],
               [2, 3, 7, 0, 5, 8, 1, 4, 6],
               [1, 4, 7, 0, 5, 6, 2, 3, 8],
               [2, 5, 8, 1, 3, 6, 0, 4, 7],
               [1, 4, 8, 0, 5, 7, 2, 3, 6]]

NUM_TRIAL_SETS = len(TRIAL_ORDER)




def create_dataset(train_type, alignment_type):

    R_W1, R_W2, R_W3 = [], [], []
    R_CW1, R_CW2, R_CW3 = [], [], []
    R_M1, R_M2, R_M3 = [], [], []

    for trial_set in range(len(TRIAL_ORDER)):
        if trial_set == NUM_TRIAL_SETS: break
        directory = "./test/R_"+str(trial_set + 1)

        print(directory+"/M_count_"+str(TRIAL_ORDER[trial_set][6])+"_sensor_29.5.pkl")
        R_M1.append(load_pickle(directory+"/M_count_"+str(TRIAL_ORDER[trial_set][6])+"_sensor_29.5.pkl"))
        R_M2.append(load_pickle(directory+"/M_count_"+str(TRIAL_ORDER[trial_set][7])+"_sensor_29.5.pkl"))
        R_M3.append(load_pickle(directory+"/M_count_"+str(TRIAL_ORDER[trial_set][8])+"_sensor_29.5.pkl"))
        R_W1.append(load_pickle(directory+"/W_count_"+str(TRIAL_ORDER[trial_set][0])+"_sensor_29.5.pkl"))
        R_W2.append(load_pickle(directory+"/W_count_"+str(TRIAL_ORDER[trial_set][1])+"_sensor_29.5.pkl"))
        R_W3.append(load_pickle(directory+"/W_count_"+str(TRIAL_ORDER[trial_set][2])+"_sensor_29.5.pkl"))
        R_CW1.append(load_pickle(directory+"/CW_count_"+str(TRIAL_ORDER[trial_set][3])+"_sensor_29.5.pkl"))
        R_CW2.append(load_pickle(directory+"/CW_count_"+str(TRIAL_ORDER[trial_set][4])+"_sensor_29.5.pkl"))
        R_CW3.append(load_pickle(directory+"/CW_count_"+str(TRIAL_ORDER[trial_set][5])+"_sensor_29.5.pkl"))


    x_full = R_CW1[0][:, 0]
    #match_set_LOW = R_CW3[0][250:1250,2]
    #match_set_HIGH = R_W1[0][280:1280,2]

    match_set_W = load_pickle("./test/R_1/W_count_"+str(TRIAL_ORDER[0][0])+"_sensor_29.5.pkl")[184:1184,2]
    match_set_CW =  load_pickle("./test/R_1/CW_count_"+str(TRIAL_ORDER[0][3])+"_sensor_29.5.pkl")[32:1032,2]
    match_set_M =  load_pickle("./test/R_1/M_count_"+str(TRIAL_ORDER[0][6])+"_sensor_29.5.pkl")[130:1130,2]


    print(R_M1[0][:,4])

    data_dict = {}


    for trial_set in range(NUM_TRIAL_SETS):


        if train_type == 'all_active' or train_type == 'noCW_active':


            if alignment_type == 'firstsec':
                data_dict['trial_set_'+str(trial_set+1)+'_X'] = np.vstack((
                lib_get_subset.get_x_matched_set(x_full, match_set_M, R_M1[trial_set][:, 1:3], is_heated=True),
                lib_get_subset.get_x_matched_set(x_full, match_set_M, R_M2[trial_set][:, 1:3], is_heated=True),
                lib_get_subset.get_x_matched_set(x_full, match_set_M, R_M3[trial_set][:, 1:3], is_heated=True),
                lib_get_subset.get_x_matched_set(x_full, match_set_W, R_W1[trial_set][:, 1:3], is_heated=True),
                lib_get_subset.get_x_matched_set(x_full, match_set_W, R_W2[trial_set][:, 1:3], is_heated=True),
                lib_get_subset.get_x_matched_set(x_full, match_set_W, R_W3[trial_set][:, 1:3], is_heated=True),
                lib_get_subset.get_x_matched_set(x_full, match_set_CW, R_CW1[trial_set][:, 1:3], is_heated=True),
                lib_get_subset.get_x_matched_set(x_full, match_set_CW, R_CW2[trial_set][:, 1:3], is_heated=True),
                lib_get_subset.get_x_matched_set(x_full, match_set_CW, R_CW3[trial_set][:, 1:3], is_heated=True)))

            elif alignment_type == 'curvemax':
                data_dict['trial_set_'+str(trial_set+1)+'_X'] = np.vstack((
                lib_get_subset.get_x_matched_deriv_set(x_full, R_M1[trial_set][:, 1:3], is_heated=True),
                lib_get_subset.get_x_matched_deriv_set(x_full, R_M2[trial_set][:, 1:3], is_heated=True),
                lib_get_subset.get_x_matched_deriv_set(x_full, R_M3[trial_set][:, 1:3], is_heated=True),
                lib_get_subset.get_x_matched_deriv_set(x_full, R_W1[trial_set][:, 1:3], is_heated=True),
                lib_get_subset.get_x_matched_deriv_set(x_full, R_W2[trial_set][:, 1:3], is_heated=True),
                lib_get_subset.get_x_matched_deriv_set(x_full, R_W3[trial_set][:, 1:3], is_heated=True),
                lib_get_subset.get_x_matched_deriv_set(x_full, R_CW1[trial_set][:, 1:3], is_heated=True),
                lib_get_subset.get_x_matched_deriv_set(x_full, R_CW2[trial_set][:, 1:3], is_heated=True),
                lib_get_subset.get_x_matched_deriv_set(x_full, R_CW3[trial_set][:, 1:3], is_heated=True)))


            elif alignment_type == 'force':
                data_dict['trial_set_'+str(trial_set+1)+'_X'] = np.vstack((
                lib_get_subset.get_x_matched_force_set(x_full, R_M1[trial_set][:, 1:5], is_heated=True),
                lib_get_subset.get_x_matched_force_set(x_full, R_M2[trial_set][:, 1:5], is_heated=True),
                lib_get_subset.get_x_matched_force_set(x_full, R_M3[trial_set][:, 1:5], is_heated=True),
                lib_get_subset.get_x_matched_force_set(x_full, R_W1[trial_set][:, 1:5], is_heated=True),
                lib_get_subset.get_x_matched_force_set(x_full, R_W2[trial_set][:, 1:5], is_heated=True),
                lib_get_subset.get_x_matched_force_set(x_full, R_W3[trial_set][:, 1:5], is_heated=True),
                lib_get_subset.get_x_matched_force_set(x_full, R_CW1[trial_set][:, 1:5], is_heated=True),
                lib_get_subset.get_x_matched_force_set(x_full, R_CW2[trial_set][:, 1:5], is_heated=True),
                lib_get_subset.get_x_matched_force_set(x_full, R_CW3[trial_set][:, 1:5], is_heated=True)))



        elif train_type == 'all_active_passive' or train_type == 'noCW_active_passive':

            if alignment_type == 'firstsec':
                data_dict['trial_set_'+str(trial_set+1)+'_X'] = np.vstack((
                lib_get_subset.get_x_matched_set(x_full, match_set_M, R_M1[trial_set][:, 1:3], is_heated=True)+lib_get_subset.get_x_matched_set(x_full, match_set_M, R_M1[trial_set][:, 1:3], is_heated=False),
                lib_get_subset.get_x_matched_set(x_full, match_set_M, R_M2[trial_set][:, 1:3], is_heated=True)+lib_get_subset.get_x_matched_set(x_full, match_set_M, R_M2[trial_set][:, 1:3], is_heated=False),
                lib_get_subset.get_x_matched_set(x_full, match_set_M, R_M3[trial_set][:, 1:3], is_heated=True)+lib_get_subset.get_x_matched_set(x_full, match_set_M, R_M3[trial_set][:, 1:3], is_heated=False),
                lib_get_subset.get_x_matched_set(x_full, match_set_W, R_W1[trial_set][:, 1:3], is_heated=True)+lib_get_subset.get_x_matched_set(x_full, match_set_W, R_W1[trial_set][:, 1:3], is_heated=False),
                lib_get_subset.get_x_matched_set(x_full, match_set_W, R_W2[trial_set][:, 1:3], is_heated=True)+lib_get_subset.get_x_matched_set(x_full, match_set_W, R_W2[trial_set][:, 1:3], is_heated=False),
                lib_get_subset.get_x_matched_set(x_full, match_set_W, R_W3[trial_set][:, 1:3], is_heated=True)+lib_get_subset.get_x_matched_set(x_full, match_set_W, R_W3[trial_set][:, 1:3], is_heated=False),
                lib_get_subset.get_x_matched_set(x_full, match_set_CW, R_CW1[trial_set][:, 1:3], is_heated=True)+lib_get_subset.get_x_matched_set(x_full, match_set_CW, R_CW1[trial_set][:, 1:3], is_heated=False),
                lib_get_subset.get_x_matched_set(x_full, match_set_CW, R_CW2[trial_set][:, 1:3], is_heated=True)+lib_get_subset.get_x_matched_set(x_full, match_set_CW, R_CW2[trial_set][:, 1:3], is_heated=False),
                lib_get_subset.get_x_matched_set(x_full, match_set_CW, R_CW3[trial_set][:, 1:3], is_heated=True)+lib_get_subset.get_x_matched_set(x_full, match_set_CW, R_CW3[trial_set][:, 1:3], is_heated=False)))

            elif alignment_type == 'curvemax':
                data_dict['trial_set_'+str(trial_set+1)+'_X'] = np.vstack((
                lib_get_subset.get_x_matched_deriv_set(x_full, R_M1[trial_set][:, 1:3], is_heated=True)+lib_get_subset.get_x_matched_deriv_set(x_full, R_M1[trial_set][:, 1:3], is_heated=False),
                lib_get_subset.get_x_matched_deriv_set(x_full, R_M2[trial_set][:, 1:3], is_heated=True)+lib_get_subset.get_x_matched_deriv_set(x_full, R_M2[trial_set][:, 1:3], is_heated=False),
                lib_get_subset.get_x_matched_deriv_set(x_full, R_M3[trial_set][:, 1:3], is_heated=True)+lib_get_subset.get_x_matched_deriv_set(x_full, R_M3[trial_set][:, 1:3], is_heated=False),
                lib_get_subset.get_x_matched_deriv_set(x_full, R_W1[trial_set][:, 1:3], is_heated=True)+lib_get_subset.get_x_matched_deriv_set(x_full, R_W1[trial_set][:, 1:3], is_heated=False),
                lib_get_subset.get_x_matched_deriv_set(x_full, R_W2[trial_set][:, 1:3], is_heated=True)+lib_get_subset.get_x_matched_deriv_set(x_full, R_W2[trial_set][:, 1:3], is_heated=False),
                lib_get_subset.get_x_matched_deriv_set(x_full, R_W3[trial_set][:, 1:3], is_heated=True)+lib_get_subset.get_x_matched_deriv_set(x_full, R_W3[trial_set][:, 1:3], is_heated=False),
                lib_get_subset.get_x_matched_deriv_set(x_full, R_CW1[trial_set][:, 1:3], is_heated=True)+lib_get_subset.get_x_matched_deriv_set(x_full, R_CW1[trial_set][:, 1:3], is_heated=False),
                lib_get_subset.get_x_matched_deriv_set(x_full, R_CW2[trial_set][:, 1:3], is_heated=True)+lib_get_subset.get_x_matched_deriv_set(x_full, R_CW2[trial_set][:, 1:3], is_heated=False),
                lib_get_subset.get_x_matched_deriv_set(x_full, R_CW3[trial_set][:, 1:3], is_heated=True)+lib_get_subset.get_x_matched_deriv_set(x_full, R_CW3[trial_set][:, 1:3], is_heated=False)))


        #print(data_dict['trial_set_1_X'].shape)


        #print(data_dict['trial_set_'+str(trial_set+1)+'_X'].shape)

        #data_dict['trial_set_'+str(trial_set+1)+'_y'] = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0])
        data_dict['trial_set_'+str(trial_set+1)+'_y'] = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])


        #print(data_dict['trial_set_'+str(trial_set+1)+'_y'].shape)



    if train_type == 'all_active':# or train_type == 'all_active_passive':
        #data_dict = lib_overlapping.get_most_overlapping(x_full, data_dict, num_pairs = 10)
        #data_dict = lib_overlapping.get_overlapping_reference(x_full, data_dict, num_referenced = 20)
        data_dict = lib_overlapping.get_meanstd_overlapping2(data_dict) #x_full,



    cross_val_dict = {}
    for trial_set in range(NUM_TRIAL_SETS):

        X_train = []
        y_train = []
        for other_trial_set in range(NUM_TRIAL_SETS):
            if other_trial_set != trial_set:

                if train_type == 'all_active' or train_type == 'all_active_passive':
                    try:
                        X_train = np.concatenate((X_train, data_dict['trial_set_'+str(other_trial_set+1)+'_X']), axis = 0)
                    except:
                        X_train = data_dict['trial_set_'+str(other_trial_set+1)+'_X']
                    try:
                        y_train = np.concatenate((y_train, data_dict['trial_set_'+str(other_trial_set+1)+'_y']), axis = 0)
                    except:
                        y_train = data_dict['trial_set_'+str(other_trial_set+1)+'_y']



                elif train_type == 'noCW_active' or train_type == 'noCW_active_passive':
                    try:
                        X_train = np.concatenate((X_train, data_dict['trial_set_' + str(other_trial_set + 1) + '_X'][0:6, :]), axis=0)
                    except:
                        X_train = data_dict['trial_set_' + str(other_trial_set + 1) + '_X'][0:6, :]
                    try:
                        y_train = np.concatenate((y_train, data_dict['trial_set_' + str(other_trial_set + 1) + '_y'][0:6]), axis=0)
                    except:
                        y_train = data_dict['trial_set_' + str(other_trial_set + 1) + '_y'][0:6]

        cross_val_dict['fold'+str(trial_set+1)+'_X_train'] = X_train #shape is (8x9) x (1000)
        cross_val_dict['fold'+str(trial_set+1)+'_y_train'] = y_train #shape is (8x9)
        cross_val_dict['fold'+str(trial_set+1)+'_X_test'] = data_dict['trial_set_'+str(trial_set+1)+'_X'] #shape is (1x9) x (1000)
        cross_val_dict['fold'+str(trial_set+1)+'_y_test'] = data_dict['trial_set_'+str(trial_set+1)+'_y'] #shape is (1x9)

    print("created dataset")
    return cross_val_dict, data_dict



def run_crossvalidation_all(cross_val_dict, trial_set, svm_type):
    #print("cross val on trial set", trial_set)
    scores_list = []
    X_train = cross_val_dict['fold'+str(trial_set+1)+'_X_train']
    y_train = cross_val_dict['fold'+str(trial_set+1)+'_y_train']
    X_test = cross_val_dict['fold'+str(trial_set+1)+'_X_test']
    y_test = cross_val_dict['fold'+str(trial_set+1)+'_y_test']


    #print(np.shape(X_train), np.shape(X_test), np.shape(y_train), np.shape(y_test))
    #print(X_test, y_test)
    #print(X_train, y_train)
    svc = svm.SVC(kernel=svm_type)
    #svc = svm.SVC(kernel='rbf')
    #svc = svm.SVC(kernel='poly', degree=3)
    clf = svc.fit(X_train, y_train)
    preds = svc.predict(X_test)

    #print("ground tr", y_test)
    #print("predicted", preds)

    #knn = KNeighborsClassifier(n_neighbors=10)
    #clf = knn.fit(X_train, y_train)
    #preds = knn.predict(X_test)

    cm = confusion_matrix(y_test, preds)
    scores = clf.score(X_test, y_test)

    #print("Confusion matrix created")
    #cmat = cmat + cm
    total = float(cm.sum())
    true = float(np.trace(cm))
    scores_list.append(true/total)
    #print(scores_list)

    print(y_test)
    if y_test[3] == 0: y_test[3] += 2
    if y_test[4] == 0: y_test[4] += 2
    if y_test[5] == 0: y_test[5] += 2
    preds_cm = preds.copy()
    if preds_cm[3] == 0: preds_cm[3] += 2
    if preds_cm[4] == 0: preds_cm[4] += 2
    if preds_cm[5] == 0: preds_cm[5] += 2
    cm = confusion_matrix(y_test, preds_cm)
    scores = clf.score(X_test, y_test)
    print(y_test, preds_cm)
    #print(cm)

    print(np.array(preds), cm)
    return np.array(preds), cm



def run_leave_one_block_out_crossvalidation_all(train_type, data_dict, svm_type):
    # print("cross val on trial set", trial_set)

    scores_list = []
    X_data = []
    y_data = []

    for trial_set in range(NUM_TRIAL_SETS):
        for block in range(len(data_dict['trial_set_' + str(trial_set + 1) + '_X'])):
            # print(trial_set, block)
            # X_train.append(data_dict['trial_set_'+str(trial_set+1)+'_X'][block, :])

            if train_type == 'all_active' or train_type == 'all_active_passive':
                X_data.append(data_dict['trial_set_' + str(trial_set + 1) + '_X'][block, :])
                y_data.append(data_dict['trial_set_' + str(trial_set + 1) + '_y'][block])

            elif train_type == 'noCW_active' or train_type == 'noCW_active_passive':
                if block < 6:
                    X_data.append(data_dict['trial_set_' + str(trial_set + 1) + '_X'][block, :])
                    y_data.append(data_dict['trial_set_' + str(trial_set + 1) + '_y'][block])
    print(np.shape(X_data))
    print(np.shape(y_data))
    X_data = np.array(X_data)
    y_data = np.array(y_data)

    cm_total = np.zeros((3, 3))

    # skf = StratifiedKFold(y_data, n_folds=3, shuffle=True)
    loo = LeaveOneOut(n=9)

    #block_sets_test = [[0, 1, 2], [0, 1, 5], [0, 1, 8], [0, 2, 4], [0, 2, 7], [0, 4, 5], [0, 4, 8], [0, 5, 7],[0, 7, 8],
    #                   [3, 1, 2], [3, 1, 5], [3, 1, 8], [3, 2, 4], [3, 2, 7], [3, 4, 5], [3, 4, 8], [3, 5, 7],[3, 7, 8],
    #                   [6, 1, 2], [6, 1, 5], [6, 1, 8], [6, 2, 4], [6, 2, 7], [6, 4, 5], [6, 4, 8], [6, 5, 7],[6, 7, 8]]
    block_sets_test = [[0, 3, 6], [0, 3, 7], [0, 3, 8], [0, 4, 6], [0, 4, 7], [0, 4, 8], [0, 5, 6], [0, 5, 7],[0, 5, 8],
                       [1, 3, 6], [1, 3, 7], [1, 3, 8], [1, 4, 6], [1, 4, 7], [1, 4, 8], [1, 5, 6], [1, 5, 7],[1, 5, 8],
                       [2, 3, 6], [2, 3, 7], [2, 3, 8], [2, 4, 6], [2, 4, 7], [2, 4, 8], [2, 5, 6], [2, 5, 7],[2, 5, 8]]
    block_sets_train = []
    for blocks_test in block_sets_test:
        blocks_train = []
        for i in range(9):
            if i not in blocks_test:
                blocks_train.append(i)
        block_sets_train.append(blocks_train)

    trial_set_skip = np.arange(0, 270, 9)
    print(trial_set_skip)

    for i in range(27):
        train_block_index = block_sets_train[i]
        test_block_index = block_sets_test[i]
        #print(train_block_index, test_block_index)

        train_index = []
        for idx in train_block_index:
            train_index.append(idx+trial_set_skip)
        train_index = sorted(list(np.array(train_index).flatten()))

        test_index = []
        for idx in test_block_index:
            test_index.append(idx+trial_set_skip)
        test_index = sorted(list(np.array(test_index).flatten()))


        print("TRAIN:", train_index, "TEST:", test_index)





        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        print(X_test)
        print(y_test)

        svc = svm.SVC(kernel=svm_type)
        # svc = svm.SVC(kernel='rbf')
        # svc = svm.SVC(kernel='poly', degree=3)
        clf = svc.fit(X_train, y_train)
        preds = svc.predict(X_test)

        # print(y_test, preds)

        cm = confusion_matrix(y_test, preds, labels=[0, 1, 2])

        scores = clf.score(X_test, y_test)

        # print("Confusion matrix created")
        # cmat = cmat + cm
        total = float(cm.sum())
        true = float(np.trace(cm))
        scores_list.append(true / total)
        # print(scores_list)

        scores = clf.score(X_test, y_test)

        #print(test_index, y_test, np.array(preds))
        cm_total += cm
        print(cm)
        print(cm_total)

    print(cm_total.T)
    print(cm_total.T/9)
    #cm_total.T
    return scores, cm_total


def run_kfold_crossvalidation_all(train_type, data_dict, svm_type):
    #print("cross val on trial set", trial_set)

    scores_list = []
    X_data = []
    y_data = []


    for trial_set in range(NUM_TRIAL_SETS):
        for block in range(len(data_dict['trial_set_' + str(trial_set + 1) + '_X'])):
            #print(trial_set, block)
            #X_train.append(data_dict['trial_set_'+str(trial_set+1)+'_X'][block, :])

            if train_type == 'all_active' or train_type == 'all_active_passive':
                X_data.append(data_dict['trial_set_' + str(trial_set + 1) + '_X'][block, :])
                y_data.append(data_dict['trial_set_' + str(trial_set + 1) + '_y'][block])

            elif train_type == 'noCW_active' or train_type == 'noCW_active_passive':
                if block < 6:
                    X_data.append(data_dict['trial_set_' + str(trial_set + 1) + '_X'][block, :])
                    y_data.append(data_dict['trial_set_' + str(trial_set + 1) + '_y'][block])
    print(np.shape(X_data))
    print(np.shape(y_data))
    X_data = np.array(X_data)
    y_data = np.array(y_data)

    cm_total = np.zeros((3,3))

    #skf = StratifiedKFold(y_data, n_folds=3, shuffle=True)
    loo = LeaveOneOut()



    #print(len(skf))
    #for train_index, test_index in skf:
    for train_index, test_index in loo.split(X_data):

        print(loo.get_n_splits(X_data), 'N SPLITS')
        print("TRAIN:", train_index, "TEST:", test_index)



        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        #print(X_test)
        #print(y_test)

        svc = svm.SVC(kernel=svm_type)
        #svc = svm.SVC(kernel='rbf')
        #svc = svm.SVC(kernel='poly', degree=3)
        clf = svc.fit(X_train, y_train)
        preds = svc.predict(X_test)

        #print(y_test, preds)

        cm = confusion_matrix(y_test, preds, labels=[0, 1, 2])


        scores = clf.score(X_test, y_test)

        #print("Confusion matrix created")
        #cmat = cmat + cm
        total = float(cm.sum())
        true = float(np.trace(cm))
        scores_list.append(true/total)
        #print(scores_list)


        scores = clf.score(X_test, y_test)


        print(test_index, y_test, np.array(preds))
        cm_total += cm
        print(cm_total)

    print(cm_total.T)
    return scores, cm_total



def split_results(results):


    M_results = np.copy(results[:, 0:3])
    M_results[M_results > 0] = 1
    M_results = 1 - M_results
    W_results = np.copy(results[:, 3:6])
    W_results[W_results > 1] = 0
    W_results[W_results < 1] = 0
    CW_results = np.copy(results[:, 6:9])
    CW_results[CW_results < 2] = 0
    CW_results[CW_results == 2] = 1
    CW_results = 1 - CW_results

    return M_results, W_results, CW_results

def plot_confusion_matrix(cm, classes, modality, path, train_type, normalize=False, title='CM: leave one-trial-out', cmap=pp.cm.Blues):
    if train_type == 'noCW_active' or train_type == 'noCW_active_passive':
        cm = cm[0:2, :]
        classes_pred = classes[0:2]

    else:
        classes_pred = classes

    """ This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        #cm = cm.astype('int')
        cm = cm.astype('float')
        print('Confusion matrix, without normalization')
    f, axarr = pp.subplots(1,1)

    print(cm)
    cax = axarr.imshow(cm, interpolation='nearest', cmap=cmap)
    axarr.set_title(title, fontsize=16)
    cbar = f.colorbar(cax)
    tick_marks = np.arange(len(classes))
    axarr.set_xticks(tick_marks)

    print(classes, classes_pred, tick_marks)

    axarr.set_xticklabels(classes, rotation=45)
    axarr.set_yticks(tick_marks[0:len(classes_pred)])
    axarr.set_yticklabels(classes_pred)
    #fmt = '.3f' if normalize else 'd'
    fmt = '.3f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        print(i, j, cm[i, j])
        axarr.text(j, i, format(cm[i, j], fmt),
                   horizontalalignment="center",
                   color="white" if cm[i, j] > thresh else "black", fontsize=14)
    pp.tight_layout()
    axarr.set_xlabel('True label', fontsize=14)
    axarr.set_ylabel('Predicted label', fontsize=14)
    savepath = path + modality + '_svm.png'
    f.savefig(savepath)

if __name__ == '__main__':

    print("Results: no cold wood in training data, only active sensor data")
    train_type = 'noCW_active'
    svm_type = 'linear'
    leave_out = 'one_trial'
    alignment_type = 'curvemax'

    cm_array = np.array([[504.,  464.],
                         [496.,  536.]])



    cm_array[0:2, 0] /= np.sum(cm_array[0:2, 0])
    cm_array[0:2, 1] /= np.sum(cm_array[0:2, 1])
    #cm_array[0:3, 2] /= np.sum(cm_array[0:3, 2])

    print(cm_array, "CM ARRAY")



    plot_confusion_matrix(cm_array, ['M', 'CW'], 'most_overlap_10_pairs_3fold_50seed',
                          './', train_type='all_active',
                          normalize=False,
                          title='CM: 10 most overlapping pairs, 3 fold 50 seed')

    print("GOT HERE")



    cross_val_dict, data_dict = create_dataset(train_type, alignment_type=alignment_type)
    cm_total = np.zeros((3,3))
    results = np.zeros((NUM_TRIAL_SETS, len(run_crossvalidation_all(cross_val_dict, 0, svm_type)[0])))


    print("GOT HERE")

    #print(cross_val_dict.shape)

    if leave_out == 'one_trial':
        for trial_set in range(NUM_TRIAL_SETS):
            results[trial_set, :], cm = run_crossvalidation_all(cross_val_dict, trial_set, svm_type)
            cm_total += cm

    elif leave_out == 'one_block':
        results, cm_total = run_kfold_crossvalidation_all(train_type, data_dict, svm_type)

    #print(cm_total, cm)


    #print(cm_total)
    plot_confusion_matrix(cm_total.T, ['M', 'W', 'CW'], 'only_roomtemp_active_'+svm_type, './', train_type, title='CM: train only roomtemp, only active sensor, svm '+svm_type)
    print(results)


    M_results, W_results, CW_results = split_results(results)
    print("percent room temp metal as metal", float(np.count_nonzero(M_results)) / (np.shape(results[:, 0:3])[0] * np.shape(results[:, 0:3])[1]))
    print("percent   room temp wood as wood", float(np.count_nonzero(W_results)) / (np.shape(results[:, 3:6])[0] * np.shape(results[:, 3:6])[1]))
    print("percent       cold wood as metal", float(np.count_nonzero(CW_results)) / (np.shape(results[:, 6:9])[0] * np.shape(results[:, 6:9])[1]))





    print("Results: no cold wood in training data, active and passive sensor data")
    train_type = 'noCW_active_passive'
    cross_val_dict, data_dict = create_dataset(train_type, alignment_type=alignment_type)
    cm_total = np.zeros((3,3))
    results = np.zeros((NUM_TRIAL_SETS, len(run_crossvalidation_all(cross_val_dict, 0, svm_type)[0])))
    for trial_set in range(NUM_TRIAL_SETS):
        print('beginning')
        results[trial_set, :], cm = run_crossvalidation_all(cross_val_dict, trial_set, svm_type)
        cm_total += cm

    #print(cm_total)
    plot_confusion_matrix(cm_total.T, ['M', 'W', 'CW'], 'only_roomtemp_active_passive_'+svm_type, './', train_type, title='CM: train only roomtemp, active and passive, svm '+svm_type)
    print(results)


    M_results, W_results, CW_results = split_results(results)
    print("percent room temp metal as metal", float(np.count_nonzero(M_results)) / (np.shape(results[:, 0:3])[0] * np.shape(results[:, 0:3])[1]))
    print("percent   room temp wood as wood", float(np.count_nonzero(W_results)) / (np.shape(results[:, 3:6])[0] * np.shape(results[:, 3:6])[1]))
    print("percent       cold wood as metal", float(np.count_nonzero(CW_results)) / (np.shape(results[:, 6:9])[0] * np.shape(results[:, 6:9])[1]))






    print("Results: all active sensor data.")
    train_type = 'all_active'
    leave_out = 'one_block'
    cross_val_dict, data_dict = create_dataset(train_type, alignment_type=alignment_type)#'force')

    cm_total = np.zeros((3,3))
    results = np.zeros((NUM_TRIAL_SETS, len(run_crossvalidation_all(cross_val_dict, 0, svm_type)[0])))

    if leave_out == 'one_trial':
        for trial_set in range(NUM_TRIAL_SETS):
            results[trial_set, :], cm = run_crossvalidation_all(cross_val_dict, trial_set, svm_type)
            cm_total += cm

    elif leave_out == 'k_fold':
        cm_list = []
        for ct in range(50):
            results, cm_total = run_kfold_crossvalidation_all(train_type, data_dict, svm_type)
            cm_list.append(cm_total.T)
        cm_sum = np.sum(np.array(cm_list).astype(float), axis = 0)
        print(cm_sum/50)

    elif leave_out == 'one_block':
        cm_list = []
        results, cm_total = run_leave_one_block_out_crossvalidation_all(train_type, data_dict, svm_type)

    #print(cm_total)
    plot_confusion_matrix(cm_total.T, ['M', 'W', 'CW'], 'all_active_'+svm_type, './', train_type, title='CM: leave-one-trial-out svm '+svm_type)#all materials, only active sensor, svm '+svm_type)
    print(results)
    M_results, W_results, CW_results = split_results(results)

    print("percent room temp metal as metal", float(np.count_nonzero(M_results)) / (np.shape(results[:, 0:3])[0] * np.shape(results[:, 0:3])[1]))
    print("percent   room temp wood as wood", float(np.count_nonzero(W_results)) / (np.shape(results[:, 3:6])[0] * np.shape(results[:, 3:6])[1]))
    print("percent       cold wood as metal", float(np.count_nonzero(CW_results)) / (np.shape(results[:, 6:9])[0] * np.shape(results[:, 6:9])[1]))






    print("Results: both active and passive sensor data.")
    train_type='all_active_passive'
    #leave_out = 'one_block'
    cross_val_dict, data_dict = create_dataset(train_type, alignment_type=alignment_type)

    cm_total = np.zeros((3,3))
    results = np.zeros((NUM_TRIAL_SETS, len(run_crossvalidation_all(cross_val_dict, 0, svm_type)[0])))

    if leave_out == 'one_trial':
        for trial_set in range(NUM_TRIAL_SETS):
            results[trial_set, :], cm = run_crossvalidation_all(cross_val_dict, trial_set, svm_type)
            cm_total += cm


    elif leave_out == 'one_block':
        cm_list = []
        results, cm_total = run_leave_one_block_out_crossvalidation_all(train_type, data_dict, svm_type)

    print(cm_total)
    plot_confusion_matrix(cm_total.T, ['M', 'W', 'CW'], 'all_passive_active_'+svm_type, '/home/henry/git/thermal_sensing/test/plus6_newwood/', train_type, title='CM: all materials, active and passive, svm '+svm_type)
    print(results)

    M_results, W_results, CW_results = split_results(results)
    print("percent room temp metal as metal", float(np.count_nonzero(M_results)) / (np.shape(results[:, 0:3])[0] * np.shape(results[:, 0:3])[1]))
    print("percent   room temp wood as wood", float(np.count_nonzero(W_results)) / (np.shape(results[:, 3:6])[0] * np.shape(results[:, 3:6])[1]))
    print("percent       cold wood as metal", float(np.count_nonzero(CW_results)) / (np.shape(results[:, 6:9])[0] * np.shape(results[:, 6:9])[1]))

