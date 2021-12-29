#!/usr/bin/env python

import os
import sys
import serial
import math, numpy as np
import roslib#; roslib.load_manifest('hrl_fabric_based_tactile_sensor')
import hrl_lib.util as ut
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.ndimage.filters import gaussian_filter1d

#from  hrl_fabric_based_tactile_sensor.map_thermistor_to_temperature import temperature

def get_derivative(x, y):

    # Get a function that evaluates the linear spline at any x
    f = InterpolatedUnivariateSpline(x, y, k=1)

    # Get a function that evaluates the derivative of the linear spline at any x
    dfdx = f.derivative()

    # Evaluate the derivative dydx at each x location...
    dydx = dfdx(x)

    dydx = gaussian_filter1d(dydx, 10.0)
    return dydx


def get_x_matched_set(x_full, match_set, new_set, is_heated):

    new_set_unheated = new_set[:, 0]
    new_set = new_set[:, 1]

    matching_bounds = [0, 100]
    x_part = x_full[matching_bounds[0]: matching_bounds[1]]
    match_set_part = match_set[matching_bounds[0]: matching_bounds[1]]

    match_set_deriv = np.array(get_derivative(x_part, match_set_part))

    new_set_deriv = np.array(get_derivative(x_full, new_set))

    summation_list = []

    for window_ct in range(500):
        summation_list.append(np.sum(np.abs(new_set_deriv[window_ct: window_ct+matching_bounds[1]] - match_set_deriv)))

    start_optim = np.argmin(summation_list)
    optim_set_heated = new_set[start_optim : start_optim + 1000]


    optim_set_unheated = new_set_unheated[start_optim : start_optim + 1000]

    optim_set_heated = optim_set_heated.tolist()
    optim_set_unheated = optim_set_unheated.tolist()

    while len(optim_set_heated) < 1000:
        optim_set_heated.append(0)

    while len(optim_set_unheated) < 1000:
        optim_set_unheated.append(0)

    optim_set_heated = np.array(optim_set_heated)
    optim_set_unheated = np.array(optim_set_unheated)

    if is_heated == True:
        optim_set = optim_set_heated
    else:
        optim_set = optim_set_unheated

    return optim_set



def get_x_matched_deriv_set(x_full, new_set, is_heated):
    print(np.shape(x_full), "x full shape")
    new_set_unheated = new_set[:, 0]
    new_set = new_set[:, 1]

    new_set_deriv = np.array(get_derivative(x_full, new_set))

    new_set_double_deriv = np.array(get_derivative(x_full, new_set_deriv))


    start_optim = np.argmin(np.abs(new_set_deriv[0:500]))


    print new_set[start_optim - 1: start_optim + 4],

    if new_set[start_optim - 2] > new_set[start_optim]: print "need reverse shift"
    elif new_set[start_optim - 1] > new_set[start_optim]: print "need reverse shift"

    if new_set[start_optim + 2] > new_set[start_optim]: print "need double shift"
    elif new_set[start_optim + 1] > new_set[start_optim]: print "need single shift"
    else: print

    optim_set_heated = new_set[start_optim : start_optim + 1000]
    optim_set_unheated = new_set_unheated[start_optim : start_optim + 1000]

    optim_set_heated = optim_set_heated.tolist()
    optim_set_unheated = optim_set_unheated.tolist()

    while len(optim_set_heated) < 1000:
        optim_set_heated.append(0)

    while len(optim_set_unheated) < 1000:
        optim_set_unheated.append(0)


    optim_set_heated = np.array(optim_set_heated)
    optim_set_unheated = np.array(optim_set_unheated)


    if is_heated == True:
        optim_set = optim_set_heated
        optim_set = list(np.array(optim_set) + 29.5 - optim_set[0])
    else:
        optim_set = optim_set_unheated

    print(np.shape(optim_set), 'optim set')
    return optim_set


def get_x_matched_force(x_full, new_set, is_heated):

    new_set_unheated = new_set[:, 0]

    force = new_set[:, 3]

    start_optim = 0
    for item in list(force)[30:300]:

        if item > 1: break
        start_optim += 1


    new_set = new_set[:, 1]


    optim_set_heated = new_set[start_optim : start_optim + 1000]
    optim_set_unheated = new_set_unheated[start_optim : start_optim + 1000]

    optim_set_heated = optim_set_heated.tolist()
    optim_set_unheated = optim_set_unheated.tolist()

    while len(optim_set_heated) < 1000:
        optim_set_heated.append(0)

    while len(optim_set_unheated) < 1000:
        optim_set_unheated.append(0)

    optim_set_heated = np.array(optim_set_heated)
    optim_set_unheated = np.array(optim_set_unheated)

    if is_heated == True:
        optim_set = optim_set_heated
        #optim_set = list(np.array(optim_set) + 29.5 - optim_set[0])
    else:
        optim_set = optim_set_unheated


    #print optim_set[0], 29.5 - optim_set[0]


    return optim_set



if __name__ == '__main__':


    R_W1, R_W2, R_W3 = [], [], []
    R_CW1, R_CW2, R_CW3 = [], [], []
    R_M1, R_M2, R_M3 = [], [], []

    alignment_type = 'curve_max'

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





    for trial_set in range(len(TRIAL_ORDER)):
        #if trial_set > 0:
        #    directory = "./test/FINAL/R_"+ str(trial_set + 3)
        #else:
        directory = "./test/FINAL/R_"+str(trial_set + 1)
        print directory
        R_W1.append(ut.load_pickle(directory+"/W_count_"+str(TRIAL_ORDER[trial_set][0])+"_sensor_29.5.pkl"))
        R_W2.append(ut.load_pickle(directory+"/W_count_"+str(TRIAL_ORDER[trial_set][1])+"_sensor_29.5.pkl"))
        R_W3.append(ut.load_pickle(directory+"/W_count_"+str(TRIAL_ORDER[trial_set][2])+"_sensor_29.5.pkl"))
        R_CW1.append(ut.load_pickle(directory+"/CW_count_"+str(TRIAL_ORDER[trial_set][3])+"_sensor_29.5.pkl"))
        R_CW2.append(ut.load_pickle(directory+"/CW_count_"+str(TRIAL_ORDER[trial_set][4])+"_sensor_29.5.pkl"))
        R_CW3.append(ut.load_pickle(directory+"/CW_count_"+str(TRIAL_ORDER[trial_set][5])+"_sensor_29.5.pkl"))
        R_M1.append(ut.load_pickle(directory+"/M_count_"+str(TRIAL_ORDER[trial_set][6])+"_sensor_29.5.pkl"))
        R_M2.append(ut.load_pickle(directory+"/M_count_"+str(TRIAL_ORDER[trial_set][7])+"_sensor_29.5.pkl"))
        R_M3.append(ut.load_pickle(directory+"/M_count_"+str(TRIAL_ORDER[trial_set][8])+"_sensor_29.5.pkl"))
        if trial_set == 29: break

    fig = plt.figure()

    x = R_CW1[0][200:1200,0]
    x_full = R_CW1[0][:, 0]
    x_full_long = R_CW1[0][:, 0]
    #match_set_LOW = R_CW3[0][150:1150,2]
    #match_set_HIGH = R_W1[0][280:1280,2]
    #match_set_CW = R_CW2[0][93:1093,2]
    #match_set_CW = R_CW1[0][32:1032,2]
    #match_set_M = R_M1[0][130:1130,2]
    #match_set_M = R_M2[0][103:1103,2]
    #match_set_M = R_M3[0][153:1153,2]
    #match_set_W = R_W1[0][184:1184,2]
    match_set_W = ut.load_pickle("./test/FINAL/R_1/W_count_"+str(TRIAL_ORDER[0][0])+"_sensor_29.5.pkl")[184:1184,2]
    match_set_CW =  ut.load_pickle("./test/FINAL/R_1/CW_count_"+str(TRIAL_ORDER[0][3])+"_sensor_29.5.pkl")[32:1032,2]
    match_set_M =  ut.load_pickle("./test/FINAL/R_1/M_count_"+str(TRIAL_ORDER[0][6])+"_sensor_29.5.pkl")[130:1130,2]

    #match_set_W = R_W3[0][175:1175,2]

    print match_set_W


    ax=[]
    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='red', label='wood')
    blue_patch = mpatches.Patch(color='blue', label='alum')
    for trial_set in range(15):
        ax.append(trial_set)
        print "trial set num: ", trial_set

        is_heated = True

        if trial_set >= 3: x_full = x_full_long

        ax[-1] = fig.add_subplot(3, 5, trial_set+1)

        if alignment_type == 'first_sec':
            ax[-1].plot(x, get_x_matched_set(x_full, match_set_W, R_W1[trial_set][:, 1:3], is_heated = is_heated), 'r-')
            ax[-1].plot(x, get_x_matched_set(x_full, match_set_W, R_W2[trial_set][:, 1:3], is_heated = is_heated), 'r--')
            ax[-1].plot(x, get_x_matched_set(x_full, match_set_W, R_W3[trial_set][:, 1:3], is_heated = is_heated), 'r-.')
            ax[-1].plot(x, get_x_matched_set(x_full, match_set_CW, R_CW1[trial_set][:, 1:3], is_heated = is_heated), 'r-')
            ax[-1].plot(x, get_x_matched_set(x_full, match_set_CW, R_CW2[trial_set][:, 1:3], is_heated = is_heated), 'r--')
            ax[-1].plot(x, get_x_matched_set(x_full, match_set_CW, R_CW3[trial_set][:, 1:3], is_heated = is_heated), 'r-.')
            ax[-1].plot(x, get_x_matched_set(x_full, match_set_M, R_M1[trial_set][:, 1:3], is_heated = is_heated), 'b-')
            ax[-1].plot(x, get_x_matched_set(x_full, match_set_M, R_M2[trial_set][:, 1:3], is_heated = is_heated), 'b--')
            ax[-1].plot(x, get_x_matched_set(x_full, match_set_M, R_M3[trial_set][:, 1:3], is_heated = is_heated), 'b-.')

        elif alignment_type == 'curve_max':
            #ax[-1].plot(x, get_x_matched_deriv_set(x_full, R_W1[trial_set][:, 1:3], is_heated = is_heated), 'r-')
            #ax[-1].plot(x, get_x_matched_deriv_set(x_full, R_W2[trial_set][:, 1:3], is_heated = is_heated), 'r--')
            #ax[-1].plot(x, get_x_matched_deriv_set(x_full, R_W3[trial_set][:, 1:3], is_heated = is_heated), 'r-.')
            ax[-1].plot(x, get_x_matched_deriv_set(x_full, R_CW1[trial_set][:, 1:3], is_heated = is_heated), 'r-')
            ax[-1].plot(x, get_x_matched_deriv_set(x_full, R_CW2[trial_set][:, 1:3], is_heated = is_heated), 'r--')
            ax[-1].plot(x, get_x_matched_deriv_set(x_full, R_CW3[trial_set][:, 1:3], is_heated = is_heated), 'r-.')
            ax[-1].plot(x, get_x_matched_deriv_set(x_full, R_M1[trial_set][:, 1:3], is_heated = is_heated), 'b-')
            ax[-1].plot(x, get_x_matched_deriv_set(x_full, R_M2[trial_set][:, 1:3], is_heated = is_heated), 'b--')
            ax[-1].plot(x, get_x_matched_deriv_set(x_full, R_M3[trial_set][:, 1:3], is_heated = is_heated), 'b-.')

        elif alignment_type == 'force':
            #ax[-1].plot(x, get_x_matched_force(x_full, R_W1[trial_set][:, 1:5], is_heated = is_heated), 'r-')
            #ax[-1].plot(x, get_x_matched_force(x_full, R_W2[trial_set][:, 1:5], is_heated = is_heated), 'r--')
            #ax[-1].plot(x, get_x_matched_force(x_full, R_W3[trial_set][:, 1:5], is_heated = is_heated), 'r-.')
            ax[-1].plot(x, get_x_matched_force(x_full, R_CW1[trial_set][:, 1:5], is_heated = is_heated), 'r-')
            ax[-1].plot(x, get_x_matched_force(x_full, R_CW2[trial_set][:, 1:5], is_heated = is_heated), 'r--')
            ax[-1].plot(x, get_x_matched_force(x_full, R_CW3[trial_set][:, 1:5], is_heated = is_heated), 'r-.')
            ax[-1].plot(x, get_x_matched_force(x_full, R_M1[trial_set][:, 1:5], is_heated = is_heated), 'b-')
            ax[-1].plot(x, get_x_matched_force(x_full, R_M2[trial_set][:, 1:5], is_heated = is_heated), 'b--')
            ax[-1].plot(x, get_x_matched_force(x_full, R_M3[trial_set][:, 1:5], is_heated = is_heated), 'b-.')



        if is_heated == True:
            ax[-1].set_ylim(26.0, 30.0)
            ax[-1].set_xlim(0, 5)
        else:
            ax[-1].set_ylim(16, 24)
        ax[-1].grid()
        plt.legend(handles=[red_patch, blue_patch ])

    plt.tight_layout()
    fig = plt.gcf()
    plt.show()

    #    = np.array([t, T0_data, T1_data, T3_data, F_data]).T
    #print
    #directory + "/room_temp_" + str(np.round(room_temp, 2)) + '.pkl', "opened"







            


