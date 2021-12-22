#!/usr/bin/python
import numpy as np
from scipy.stats import norm

del_0 = -0.2
del_1 = 0.0
alpha = 0.05
temp_threshold = 3.5

def run_similarity_test():

    s = open('study_data.txt', 'r').read()
    study_data_dict = eval(s)

    for item in study_data_dict:
        print('main study', item, len(study_data_dict[item]))

    #if all of a particular subject's trials are out of range, remove that subject's data from the analysis
    subj_idx = 0
    while subj_idx < len(study_data_dict['temp_discrepancy']):
        if all(j > temp_threshold for j in np.abs(study_data_dict['temp_discrepancy'][subj_idx])):
            for item in study_data_dict:
                study_data_dict[item].pop(subj_idx)
        else:
            subj_idx += 1


    for item in study_data_dict:
        print('main study', item, len(study_data_dict[item]))


    x11 = np.array(study_data_dict['M|M_and_M|CW'])
    x10 = np.array(study_data_dict['M|M_and_W|CW'])
    x01 = np.array(study_data_dict['W|M_and_M|CW'])
    x00 = np.array(study_data_dict['W|M_and_W|CW'])
    temp_discrepancy = np.array(study_data_dict['temp_discrepancy'])

    x_11k, x_10k, x_01k, x_00k = [], [], [], []


    for trial_idx in range(len(temp_discrepancy)):

        x_11k.append(0)
        x_10k.append(0)
        x_01k.append(0)
        x_00k.append(0)

        for case_idx in range(3):
            #must be less than threshold specified
            if np.abs(temp_discrepancy[trial_idx][case_idx]) <= temp_threshold:

                if x11[trial_idx][case_idx] == 1: x_11k[-1] += 1
                if x10[trial_idx][case_idx] == 1: x_10k[-1] += 1
                if x01[trial_idx][case_idx] == 1: x_01k[-1] += 1
                if x00[trial_idx][case_idx] == 1: x_00k[-1] += 1

    #print out the trial info per subject on data within the threshold
    print('M|M and M|CW:', x_11k)
    print('M|M and W|CW:', x_10k)
    print('W|M and M|CW:', x_01k)
    print('W|M and W|CW:', x_00k)

    x_11k, x_10k, x_01k, x_00k = np.array(x_11k), np.array(x_10k), np.array(x_01k), np.array(x_00k)

    n_k = np.sum((x_11k, x_10k, x_01k, x_00k), axis = 0)*1.

    p_10k = x_10k/n_k
    p_01k = x_01k/n_k


    x_1ok = x_11k + x_10k*1.
    x_o1k = x_01k + x_11k*1.

    x_1oo = np.sum(x_1ok, axis = 0)*1.
    x_o1o = np.sum(x_o1k, axis = 0)*1.

    #print x_1oo, x_o1o

    n_o = np.sum(n_k, axis = 0)

    #print n_o, 'sum'


    ph_1oo = x_1oo*1./n_o
    ph_o1o = x_o1o*1./n_o
    ph_1ok = x_1ok*1./n_k
    ph_o1k = x_o1k*1./n_k

    #difference between new and standard procedures in the proportions
    dh = ph_1oo - ph_o1o

    #number of clusters
    K = np.shape(n_k)[0]
    K = K*1.
    n = n_o/K #average cluster size

    #this is p-bar
    #pb_1oo = (ph_1oo + ph_o1o + del_0)/2.
    #pb_o1o = (ph_1oo + ph_o1o - del_0)/2.
    #pb_1ok = (ph_1ok + ph_o1k + del_0)/2.
    #pb_o1k = (ph_1ok + ph_o1k - del_0)/2.

    pb_1oo = ((x_1oo + x_o1o)/n_o + del_0)/2.
    pb_o1o = ((x_1oo + x_o1o)/n_o - del_0)/2.

    #equation 1b
    varh_ph_1oo = (K/(K-1))*np.sum(np.square(x_1ok - n_k*pb_1oo)/np.square(n_o))
    #equation 1c
    varh_ph_o1o = (K/(K-1))*np.sum(np.square(x_o1k - n_k*pb_o1o)/np.square(n_o))
    #equation 1d
    covh_ph_1oo_o1o = (K/(K-1))*np.sum((x_1ok - n_k*pb_1oo)*(x_o1k - n_k*pb_o1o)/np.square(n_o))
    #equation 1a
    varh_dh_m_del0 = varh_ph_1oo + varh_ph_o1o - 2*covh_ph_1oo_o1o
    #equation 1
    z = (dh - del_0)/np.sqrt(varh_dh_m_del0)
    print(z, 'z statistic from Nam et al')


    #z calculation from Yang's simplified version
    z_O = np.sqrt((K-1)*1./K)*((np.sum(x_10k - x_01k)) - del_0*n_o)/np.sqrt(np.sum(np.square(x_10k - x_01k - del_0*n_k)))
    print(z_O, 'Yang\'s simplified version')


    p_val = norm.sf(np.abs(z))
    print(p_val, 'one sided p value from Nam et al')




if __name__ == '__main__':

    run_similarity_test()

  
