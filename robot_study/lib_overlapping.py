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
import matplotlib.pyplot as plt


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



def get_most_overlapping(x_full, data_dict, num_pairs):

    matching_criteria = []

    for trial_set in range(len(TRIAL_ORDER)):
        optim_set = data_dict['trial_set_' + str(trial_set + 1) + '_X']
        print("trial set 1 shape", optim_set.shape)

        for block in [0, 1, 2]:
            present_set = optim_set[block, :]
            for other_trial_set in range(len(TRIAL_ORDER)):
                if other_trial_set != trial_set:
                    for CW_block in [6, 7, 8]:
                        check_set = data_dict['trial_set_' + str(other_trial_set + 1) + '_X'][CW_block, :]
                        #print(present_set)
                        #print(check_set)
                        matching_criteria.append([trial_set, block, other_trial_set, CW_block, np.sum(np.square(present_set - check_set))])

            #break


        #break

    matching_criteria = np.array(matching_criteria)

    matching_criteria = matching_criteria[matching_criteria[:,4].argsort()]
    #matching_criteria = np.flipud(matching_criteria)

    #for criteria in matching_criteria:
    #    print(criteria)



    #matching_criteria = matching_criteria[0:num_pairs, :]

    #here make sure there arne't any duplicates in the list
    reduced_matching_criteria = []
    iterator = 0
    while len(reduced_matching_criteria) < num_pairs:
        next_criteria = matching_criteria[iterator, :]
        should_append = True
        print(next_criteria)
        for item in reduced_matching_criteria:
            if item[0] == next_criteria[0] and item[1] == next_criteria[1]:
                should_append = False
            if item[2] == next_criteria[2] and item[3] == next_criteria[3]:
                should_append = False
        if should_append == True:
            reduced_matching_criteria.append(next_criteria)
        iterator += 1

    matching_criteria = np.array(reduced_matching_criteria).astype(int)





    print(matching_criteria.astype(int))

    X_data = []
    y_data = []
    #Use matching criteria indexing to create data subsets that we can use for LOOCV
    for ct in range(num_pairs):
        trial_set = int(matching_criteria[ct, 0])
        block = int(matching_criteria[ct, 1])
        X_data.append(data_dict['trial_set_' + str(trial_set + 1) + '_X'][block, :])
        y_data.append(0)
    for ct in range(num_pairs):
        trial_set = int(matching_criteria[ct, 2])
        block = int(matching_criteria[ct, 3])
        X_data.append(data_dict['trial_set_' + str(trial_set + 1) + '_X'][block, :])
        y_data.append(1)
    X_data = np.array(X_data)
    y_data = np.array(y_data)

    # print(len(skf))
    # for train_index, test_index in skf:





    cm_total = np.zeros((2,2))

    #loo = LeaveOneOut(num_pairs)
    #for train_index, test_index in loo:

    for i in range(1000):
        kfold = KFold(n_splits=3, shuffle = True)
        kfold.get_n_splits(np.arange(num_pairs))
        for train_index, test_index in kfold.split(np.arange(num_pairs)):
            print("TRAIN:", train_index, "TEST:", test_index)
            train_index_second = list(np.array(train_index) + num_pairs)
            train_index = np.array(list(train_index) + train_index_second)
            test_index_second = list(np.array(test_index) + num_pairs)
            test_index = np.array(list(test_index) + test_index_second)
            print("TRAIN:", train_index, "TEST:", test_index)

            print(np.shape(X_data))

            X_train, X_test = X_data[train_index], X_data[test_index]
            print(np.shape(X_train))

            y_train, y_test = y_data[train_index], y_data[test_index]
            # print(X_test)
            # print(y_test)

            svc = svm.SVC(kernel='linear')
            # svc = svm.SVC(kernel='rbf')
            # svc = svm.SVC(kernel='poly', degree=3)
            clf = svc.fit(X_train, y_train)
            preds = svc.predict(X_test)

            print(y_test, preds)

            cm = confusion_matrix(y_test, preds, labels=[0, 1])

            scores = clf.score(X_test, y_test)

            #print(test_index, y_test, np.array(preds))
            cm_total += cm
            print(cm)
            print(cm_total)
    print(cm_total / 1000)

    '''
    ax=[]
    fig = plt.figure()
    print(np.shape(X_data))
    for trial_set in range(num_pairs):
        ax.append(trial_set)
        print("trial set num: ", trial_set)

        ax[-1] = fig.add_subplot(4, 5, trial_set+1)


        ax[-1].plot(np.arange(1000)/200., X_data[trial_set+num_pairs, :], 'r-', linewidth=0.5)
        ax[-1].plot(np.arange(1000)/200., X_data[trial_set, :], 'b-', linewidth=0.5)

        ax[-1].set_ylim(26.0, 30.0)
        ax[-1].set_xlim(0, 5)
        ax[-1].grid()

    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    '''



def get_meanstd_overlapping2(data_dict):


    print("GOT TO OVERLAPPING2")

    X_data_all_M = []
    y_data_all_M = []
    X_data_all_CW = []
    y_data_all_CW = []
    X_data_all_W = []
    y_data_all_W = []

    for trial_set in range(len(TRIAL_ORDER)):
        optim_set = data_dict['trial_set_' + str(trial_set + 1) + '_X']
        print("trial set 1 shape", optim_set.shape)

        for block in [0, 1, 2]:
            X_data_all_M.append(optim_set[block, :])
            y_data_all_M.append(0)

        for block in [3, 4, 5]:
            X_data_all_W.append(optim_set[block, :])
            y_data_all_W.append(1)


        for block in [6, 7, 8]:
            X_data_all_CW.append(optim_set[block, :])
            y_data_all_CW.append(1)




    X_data_all_M = np.array(X_data_all_M)
    X_data_all_CW = np.array(X_data_all_CW)
    X_data_all_W = np.array(X_data_all_W)
    print(X_data_all_M.shape)
    X_data_all_M_mean = np.mean(X_data_all_M, axis = 0)
    X_data_all_M_std = np.std(X_data_all_M, axis = 0)
    X_data_all_CW_mean = np.mean(X_data_all_CW, axis = 0)
    X_data_all_CW_std = np.std(X_data_all_CW, axis = 0)
    X_data_all_W_mean = np.mean(X_data_all_W, axis = 0)
    X_data_all_W_std = np.std(X_data_all_W, axis = 0)
    #print(X_data_all_M_mean.shape)
    #print(X_data_all_M_mean)
    #print(X_data_all_M_std)

    #print(X_data_all_M_mean - X_data_all_M_std)


    X_upper_bound = np.amin([X_data_all_CW_mean + X_data_all_CW_std, X_data_all_M_mean + X_data_all_M_std], axis = 0)
    X_lower_bound = np.amax([X_data_all_CW_mean - X_data_all_CW_std, X_data_all_M_mean - X_data_all_M_std], axis = 0)
    #print("BLAH")
    #print(X_upper_bound)
    #print(X_lower_bound)
    #print("BLAH2")


    fig, ax = plt.subplots()

    #ax.plot(np.arange(0, 1000)/200., X_data_all_M_mean[0:1000], color = '#d95f02', label="mean M, active")
    #ax.plot(np.arange(0, 1000)/200., X_data_all_CW_mean[0:1000], '--', color = '#1b9e77', label="mean CW, active")
    #ax.plot(np.arange(0, 1000)/200., X_data_all_W_mean[0:1000],  '-.', color = '#7570b3', label="mean W, active")
    #ax.fill_between(np.arange(0, 1000)/200.,
    #                X_data_all_M_mean[0:1000] - X_data_all_M_std[0:1000],
    #                X_data_all_M_mean[0:1000] + X_data_all_M_std[0:1000], facecolor='#d95f02', alpha = 0.3)
    #ax.fill_between(np.arange(0, 1000)/200.,
    #                X_data_all_CW_mean[0:1000] - X_data_all_CW_std[0:1000],
    #                X_data_all_CW_mean[0:1000] + X_data_all_CW_std[0:1000], facecolor='#1b9e77', alpha = 0.3)
    #ax.fill_between(np.arange(0, 1000)/200.,
    #                X_data_all_W_mean[0:1000] - X_data_all_W_std[0:1000],
    #                X_data_all_W_mean[0:1000] + X_data_all_W_std[0:1000], facecolor='#7570b3', alpha = 0.3)
    #ax.plot(np.arange(0, 1000)/200., X_lower_bound, 'c-')
    #ax.plot(np.arange(0, 1000)/200., X_upper_bound, 'c-')

    ax.plot(np.arange(0, 1000)/200., X_data_all_M_mean[1000:2000], color = '#d95f02', label="mean M, passive")
    ax.plot(np.arange(0, 1000)/200., X_data_all_CW_mean[1000:2000], '--', color = '#1b9e77', label="mean CW, passive")
    ax.plot(np.arange(0, 1000)/200., X_data_all_W_mean[1000:2000],  '-.', color = '#7570b3', label="mean W, passive")

    ax.fill_between(np.arange(0, 1000)/200.,
                    X_data_all_M_mean[1000:2000] - X_data_all_M_std[1000:2000],
                    X_data_all_M_mean[1000:2000] + X_data_all_M_std[1000:2000], facecolor='#d95f02', alpha = 0.3)
    ax.fill_between(np.arange(0, 1000)/200.,
                    X_data_all_CW_mean[1000:2000] - X_data_all_CW_std[1000:2000],
                    X_data_all_CW_mean[1000:2000] + X_data_all_CW_std[1000:2000], facecolor='#1b9e77', alpha = 0.3)
    ax.fill_between(np.arange(0, 1000)/200.,
                    X_data_all_W_mean[1000:2000] - X_data_all_W_std[1000:2000],
                    X_data_all_W_mean[1000:2000] + X_data_all_W_std[1000:2000], facecolor='#7570b3', alpha = 0.3)
    #ax.plot(np.arange(0, 1000)/200., X_lower_bound, 'c-')
    #ax.plot(np.arange(0, 1000)/200., X_upper_bound, 'c-')



    ax.legend(loc=3, fontsize=10)
    ax.set_title("Mean Temperature and \n Standard Deviation Bounds", fontsize=16)
    ax.set_xlabel("Time Elapsed (seconds)", fontsize=14)
    ax.set_ylabel("Temperature, Celsius", fontsize=14)
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()


    X_data_all_M = list(X_data_all_M)
    X_data_all_CW = list(X_data_all_CW)

    X_data = []
    y_data = []

    #ax.plot(np.arange(0, 1000) / 200., X_data_all_mean + X_data_all_std, 'k-', label="mean")
    for ct in range(len(X_data_all_M)):

        if np.min(X_upper_bound - X_data_all_M[ct]) > 0 and\
                np.min(X_data_all_M[ct] - X_lower_bound) > 0:
            #ax.plot(np.arange(0, 1000) / 200., X_data_all[ct], 'r-', linewidth = 0.5)
            X_data.append(X_data_all_M[ct])
            y_data.append(y_data_all_M[ct])

    for ct in range(len(X_data_all_CW)):

        if np.min(X_upper_bound - X_data_all_CW[ct]) > 0 and\
                np.min(X_data_all_CW[ct] - X_lower_bound) > 0:
            #ax.plot(np.arange(0, 1000) / 200., X_data_all[ct], 'r-', linewidth = 0.5)
            X_data.append(X_data_all_CW[ct])
            y_data.append(y_data_all_CW[ct])

    print(np.shape(X_data))
    print(np.shape(y_data))
    print(np.sum(y_data))

    #ax.legend()
    #plt.tight_layout()
    #fig = plt.gcf()
    #plt.show()

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    from time import sleep
    sleep(4)

    cm_total = np.zeros((2,2))

    for i in range(1000):

        skf = StratifiedKFold(y_data, n_folds=3, shuffle=True)
        for train_index, test_index in skf:
            print("TRAIN:", train_index, "TEST:", test_index)

            print(np.shape(X_data))

            X_train = X_data[train_index]
            X_test = X_data[test_index]
            print(np.shape(X_train))

            y_train, y_test = y_data[train_index], y_data[test_index]
            # print(X_test)
            # print(y_test)

            svc = svm.SVC(kernel='linear')
            # svc = svm.SVC(kernel='rbf')
            # svc = svm.SVC(kernel='poly', degree=3)
            clf = svc.fit(X_train, y_train)
            preds = svc.predict(X_test)

            print(y_test, preds)

            cm = confusion_matrix(y_test, preds, labels=[0, 1])

            scores = clf.score(X_test, y_test)

            #print(test_index, y_test, np.array(preds))
            cm_total += cm
            print(cm)
            print(cm_total)
    print(cm_total/1000)

    '''
    ax=[]
    fig = plt.figure()
    print(np.shape(X_data))
    for trial_set in range(num_pairs):
        ax.append(trial_set)
        print("trial set num: ", trial_set)

        ax[-1] = fig.add_subplot(4, 5, trial_set+1)


        ax[-1].plot(np.arange(1000)/200., X_data[trial_set+num_pairs, :], 'r-', linewidth=0.5)
        ax[-1].plot(np.arange(1000)/200., X_data[trial_set, :], 'b-', linewidth=0.5)

        ax[-1].set_ylim(26.0, 30.0)
        ax[-1].set_xlim(0, 5)
        ax[-1].grid()

    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    '''
def get_meanstd_overlapping(x_full, data_dict):

    X_data_all = []
    y_data_all = []

    for trial_set in range(len(TRIAL_ORDER)):
        optim_set = data_dict['trial_set_' + str(trial_set + 1) + '_X']
        print("trial set 1 shape", optim_set.shape)

        for block in [0, 1, 2, 6, 7, 8]:
            X_data_all.append(optim_set[block, :])
            if block in [0, 1, 2]:
                y_data_all.append(0)
            else:
                y_data_all.append(1)

    X_data_all = np.array(X_data_all)
    print(X_data_all.shape)
    X_data_all_mean = np.mean(X_data_all, axis = 0)
    X_data_all_std = np.std(X_data_all, axis = 0)
    print(X_data_all_mean.shape)
    print(X_data_all_mean)
    print(X_data_all_std)

    print(X_data_all_mean - X_data_all_std)

    fig, ax = plt.subplots()

    ax.plot(np.arange(0, 1000)/200., X_data_all_mean, 'k-', label="mean")
    ax.fill_between(np.arange(0, 1000)/200.,
                    X_data_all_mean - X_data_all_std,
                    X_data_all_mean + X_data_all_std)
    ax.legend()
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()

    X_data_all = list(X_data_all)

    X_data = []
    y_data = []

    ax.plot(np.arange(0, 1000) / 200., X_data_all_mean + X_data_all_std, 'k-', label="mean")
    for ct in range(len(X_data_all)):

        print(ct, np.shape(X_data_all_mean + X_data_all_std - X_data_all[ct]),
              np.min(X_data_all_mean + X_data_all_std - X_data_all[ct]),
              np.min(X_data_all[ct] + X_data_all_std - X_data_all_mean))

        if np.min(X_data_all_mean + X_data_all_std - X_data_all[ct]) > 0 and\
                np.min(X_data_all[ct] - X_data_all_mean + X_data_all_std) > 0:
            #ax.plot(np.arange(0, 1000) / 200., X_data_all[ct], 'r-', linewidth = 0.5)
            X_data.append(X_data_all[ct])
            y_data.append(y_data_all[ct])

    print(np.shape(X_data))
    print(np.shape(y_data))
    print(np.sum(y_data))

    #ax.legend()
    #plt.tight_layout()
    #fig = plt.gcf()
    #plt.show()

    X_data = np.array(X_data)
    y_data = np.array(y_data)



    cm_total = np.zeros((2,2))

    for i in range(1000):

        skf = StratifiedKFold(y_data, n_folds=3, shuffle=True)
        for train_index, test_index in skf:
            print("TRAIN:", train_index, "TEST:", test_index)

            print(np.shape(X_data))

            X_train = X_data[train_index]
            X_test = X_data[test_index]
            print(np.shape(X_train))

            y_train, y_test = y_data[train_index], y_data[test_index]
            # print(X_test)
            # print(y_test)

            svc = svm.SVC(kernel='linear')
            # svc = svm.SVC(kernel='rbf')
            # svc = svm.SVC(kernel='poly', degree=3)
            clf = svc.fit(X_train, y_train)
            preds = svc.predict(X_test)

            print(y_test, preds)

            cm = confusion_matrix(y_test, preds, labels=[0, 1])

            scores = clf.score(X_test, y_test)

            #print(test_index, y_test, np.array(preds))
            cm_total += cm
            print(cm)
            print(cm_total)
    print(cm_total/1000)

    '''
    ax=[]
    fig = plt.figure()
    print(np.shape(X_data))
    for trial_set in range(num_pairs):
        ax.append(trial_set)
        print("trial set num: ", trial_set)

        ax[-1] = fig.add_subplot(4, 5, trial_set+1)


        ax[-1].plot(np.arange(1000)/200., X_data[trial_set+num_pairs, :], 'r-', linewidth=0.5)
        ax[-1].plot(np.arange(1000)/200., X_data[trial_set, :], 'b-', linewidth=0.5)

        ax[-1].set_ylim(26.0, 30.0)
        ax[-1].set_xlim(0, 5)
        ax[-1].grid()

    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    '''


def get_overlapping_reference(x_full, data_dict, num_referenced):

    matching_criteria = []

    #get set of all possible pairs of RMSE scores
    for trial_set in range(len(TRIAL_ORDER)):
        optim_set = data_dict['trial_set_' + str(trial_set + 1) + '_X']
        print("trial set 1 shape", optim_set.shape)
        for block in [0, 1, 2]:
            present_set = optim_set[block, :]
            for other_trial_set in range(len(TRIAL_ORDER)):
                if other_trial_set != trial_set:
                    for CW_block in [6, 7, 8]:
                        check_set = data_dict['trial_set_' + str(other_trial_set + 1) + '_X'][CW_block, :]
                        #print(present_set)
                        #print(check_set)
                        matching_criteria.append([trial_set, block, other_trial_set, CW_block, np.sum(np.square(present_set - check_set))])



    matching_criteria = np.array(matching_criteria)
    matching_criteria = matching_criteria[matching_criteria[:,4].argsort()]
    #matching_criteria = np.flipud(matching_criteria)


    referenced_pair = matching_criteria[0, :]
    print(referenced_pair)


    #get set of nearest metals to the metal
    matching_M_criteria = []
    M_trial_set = int(referenced_pair[0])
    M_block = int(referenced_pair[1])
    present_set = data_dict['trial_set_' + str(M_trial_set + 1) + '_X'][M_block, :]
    for other_trial_set in range(len(TRIAL_ORDER)):
        for other_block in [0, 1, 2]:
            if other_trial_set == M_trial_set and other_block == M_block: pass
            else:
                check_set = data_dict['trial_set_' + str(other_trial_set + 1) + '_X'][other_block, :]
                matching_M_criteria.append([other_trial_set, other_block, np.sum(np.square(present_set - check_set))])
    matching_M_criteria = np.array(matching_M_criteria)
    matching_M_criteria = matching_M_criteria[matching_M_criteria[:,2].argsort()]

    print(matching_M_criteria[0:num_referenced-1, :])


    #get set of nearest cold woods to the cold wood
    matching_CW_criteria = []
    CW_trial_set = int(referenced_pair[2])
    CW_block = int(referenced_pair[3])
    present_set = data_dict['trial_set_' + str(CW_trial_set + 1) + '_X'][CW_block, :]
    for other_trial_set in range(len(TRIAL_ORDER)):
        for other_block in [6, 7, 8]:
            if other_trial_set == CW_trial_set and other_block == CW_block: pass
            else:
                check_set = data_dict['trial_set_' + str(other_trial_set + 1) + '_X'][other_block, :]
                matching_CW_criteria.append([other_trial_set, other_block, np.sum(np.square(present_set - check_set))])
    matching_CW_criteria = np.array(matching_CW_criteria)
    matching_CW_criteria = matching_CW_criteria[matching_CW_criteria[:,2].argsort()]

    print(matching_CW_criteria[0:num_referenced-1, :])




    print(matching_criteria.astype(int))


    #Use matching criteria indexing to create data subsets that we can use for LOOCV
    X_data = []
    y_data = []
    trial_set = int(referenced_pair[0])
    block = int(referenced_pair[1])
    X_data.append(data_dict['trial_set_' + str(trial_set + 1) + '_X'][block, :])
    y_data.append(0)
    for ct in range(num_referenced-1):
        trial_set = int(matching_M_criteria[ct, 0])
        block = int(matching_M_criteria[ct, 1])
        X_data.append(data_dict['trial_set_' + str(trial_set + 1) + '_X'][block, :])
        y_data.append(0)


    trial_set = int(referenced_pair[2])
    block = int(referenced_pair[3])
    X_data.append(data_dict['trial_set_' + str(trial_set + 1) + '_X'][block, :])
    y_data.append(1)
    for ct in range(num_referenced-1):
        trial_set = int(matching_CW_criteria[ct, 0])
        block = int(matching_CW_criteria[ct, 1])
        X_data.append(data_dict['trial_set_' + str(trial_set + 1) + '_X'][block, :])
        y_data.append(1)

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    print(y_data)

    # print(len(skf)
    # for train_index, test_index in skf:


    cm_total = np.zeros((2,2))

    #loo = LeaveOneOut(num_pairs)
    #for train_index, test_index in loo:


    for i in range(1000):

        skf = StratifiedKFold(y_data, n_folds=3, shuffle=True)
        for train_index, test_index in skf:
            print("TRAIN:", train_index, "TEST:", test_index)

            print(np.shape(X_data))

            X_train, X_test = X_data[train_index], X_data[test_index]
            print(np.shape(X_train))

            y_train, y_test = y_data[train_index], y_data[test_index]
            # print(X_test)
            # print(y_test)

            svc = svm.SVC(kernel='linear')
            # svc = svm.SVC(kernel='rbf')
            # svc = svm.SVC(kernel='poly', degree=3)
            clf = svc.fit(X_train, y_train)
            preds = svc.predict(X_test)

            print(y_test, preds)

            cm = confusion_matrix(y_test, preds, labels=[0, 1])

            scores = clf.score(X_test, y_test)

            #print(test_index, y_test, np.array(preds))
            cm_total += cm
            print(cm)
            print(cm_total)
    print(cm_total/1000)
    #[[6.234 3.766]
    # [3.221 6.779]]



    '''
    ax=[]
    fig = plt.figure()
    print(np.shape(X_data))
    for trial_set in range(num_pairs):
        ax.append(trial_set)
        print("trial set num: ", trial_set)

        ax[-1] = fig.add_subplot(4, 5, trial_set+1)


        ax[-1].plot(np.arange(1000)/200., X_data[trial_set+num_pairs, :], 'r-', linewidth=0.5)
        ax[-1].plot(np.arange(1000)/200., X_data[trial_set, :], 'b-', linewidth=0.5)

        ax[-1].set_ylim(26.0, 30.0)
        ax[-1].set_xlim(0, 5)
        ax[-1].grid()

    plt.tight_layout()
    fig = plt.gcf()
    plt.show()
    '''