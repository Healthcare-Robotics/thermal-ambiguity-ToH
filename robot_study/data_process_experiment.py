# Process Data before running ML algorithms

import pylab as pyl
import numpy as np
import matplotlib.pyplot as pp

import scipy as scp
import scipy.ndimage as ni
from scipy.signal import butter, lfilter, freqz

import rospy
import pickle
import cPickle as pk

import unittest
import ghmm
import ghmmwrapper
import random

from sklearn import decomposition
import copy
import os, os.path

## read a pickle and return the object.
# @param filename - name of the pkl
# @return - object that had been pickled.
def load_pickle(filename):
    try:
        p = open(filename, 'r')
    except IOError:
        print "Pickle file cannot be opened."
        return None
    try:
        picklelicious = pk.load(p)
    except ValueError:
        print 'load_pickle failed once, trying again'
        p.close()
        p = open(filename, 'r')
        picklelicious = pk.load(p)

    p.close()
    return picklelicious

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5*fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b,a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Define slope features

def create_slope(Fvec, fs): 

    Fvec = np.array(Fvec, dtype=np.float)
    slope = np.gradient(Fvec, 1/fs)
            
    order = 8
    cutoff = 20
    # Filter the data
    #slope = butter_lowpass_filter(slope, cutoff, fs, order)
    return slope.tolist()
    
# Scaling function wrt to all data

def scaling(mat, features, temp_num): 

    Fvec = [[] for j in range(features)]
    mean = [0.0]*features
    std = [0.0]*features
    
    for i in range(temp_num):
        temp_len = (len(mat[i]))/features
        for j in range(features):
            Fvec[j].append(mat[i][j*temp_len:(j+1)*temp_len])
        
    for j in range(features):
        Fvec[j] = np.array(sum(Fvec[j],[]))
        mean[j] = np.mean(Fvec[j])
        std[j] = np.std(Fvec[j])
            
    for i in range(temp_num):
        temp_len = len(mat[i])
        for j in range(temp_len):
            for k in range(features):
                if (j >= k*temp_len/features and j < (k+1)*temp_len/features):
                    mat[i][j] = ((mat[i][j] - mean[k])/std[k])
    return mat

def run_pca(mat):
    pca = decomposition.PCA(n_components=100)
    pca.fit(np.array(mat))
    print sum(pca.explained_variance_ratio_)
    reduced_mat = pca.transform(np.array(mat))
    #return np.array(mat)    
    return reduced_mat
    

def interpolate_data(x, y, num_samples):
    target_time = np.linspace(0,num_samples/120.,num_samples)
    ynew = np.interp(target_time, x, y)
    return ynew.tolist()

def feature_vector(ta, temp_num, data_path, exp_list, num_subjects, features, categories_dict):
    trials_list = []
    for i in range(np.size(exp_list)):
        trials = 0
        for j in range(1, num_subjects+1):
            for category in categories_dict[exp_list[i]]:
                dir_path = data_path + exp_list[i] + 'subject' + str(j) + '/' + category
                num_trials = len([fl for fl in os.listdir(dir_path) if fl.endswith('.pkl') and os.path.isfile(os.path.join(dir_path,fl))])
                for num_trial in range(1, num_trials+1):
                    path = dir_path + np.str(num_trial)
                    #print path
                    temp_data = []
                    data_file = load_pickle(path + '.pkl')
                    time = data_file[0]
                    for k in range(1,19):
                        data_file[k] = interpolate_data(time, data_file[k], 1200)
                    if features == 'f':
                        temp_data.append(data_file[3])
                        #temp_data.append(create_slope(data_file[3],120.0))
                    elif features == '3f':
                        for idx in range(1,4):
                            temp_data.append(data_file[idx])
                    #        temp_data.append(create_slope(data_file[idx],120.0))
                    elif features == '3f_3tau':
                        for idx in range(1,7):
                            temp_data.append(data_file[idx])
                            #temp_data.append(create_slope(data_file[idx],120.0))
                    elif features == '3p':
                        for idx in range(7,10):
                            temp_data.append(data_file[idx])
                    elif features == '3p_3theta':
                        for idx in range(7,13):
                            temp_data.append(data_file[idx])
                    elif features == '3f_3p':
                        for idx in range(1,4):
                            temp_data.append(data_file[idx])
                            #temp_data.append(create_slope(data_file[idx],120.0))
                        for idx in range(7,10):
                            pos = np.array(data_file[idx])
                            pos = pos - pos[0]
                            temp_data.append(pos.tolist())
                            #temp_data.append(create_slope(pos,120.0))
                        #for idx in range(13,16):
                            #temp_data.append(data_file[idx])
                    elif features == '3f_3tau_3p_3theta':
                        for idx in range(1,13):
                            temp_data.append(data_file[idx])
                    #    for idx in range(1,7):
                    #        temp_data.append(create_slope(data_file[idx],120.0))
                    else:
                        for idx in range(1,19):
                            temp_data.append(data_file[idx])
                        #for idx in range(1,7):
                        #    temp_data.append(create_slope(data_file[idx],120.0))

                    temp_data = sum(temp_data,[])
                    ta.append(temp_data)
                temp_num = temp_num+1
                trials = trials + 1
                trials_list.append(trials)

    return ta, temp_num, trials_list

def create_feature_vector(data_path, exp_list, num_subjects, features, features_num, categories_dict):
	
    ta = []
    temp_num = 0
    ta, temp_num, trials_list = feature_vector(ta, temp_num, data_path, exp_list, num_subjects, features, categories_dict)
    print "Data loaded..."

    Fmat_original = [0.0]*temp_num

    ## Creating Feature Vector

    idx = 0
    while (idx < temp_num):
        Fmat_original[idx] = ta[idx]
        idx = idx + 1

    if features == 'f':
        pass
    else:
        Fmat_original = scaling(Fmat_original, features_num, temp_num)

    #Fmat_original = run_pca(Fmat_original)
    return Fmat_original, trials_list

