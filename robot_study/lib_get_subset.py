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
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.ndimage.filters import gaussian_filter1d
import sys


def get_derivative(x, y):

    # Get a function that evaluates the linear spline at any x
    f = InterpolatedUnivariateSpline(x, y, k=1)

    # Get a function that evaluates the derivative of the linear spline at any x
    dfdx = f.derivative()

    # Evaluate the derivative dydx at each x location...
    dydx = dfdx(x)

    dydx = gaussian_filter1d(dydx, 10.0)
    return dydx


def get_x_matched_deriv_set(x_full, new_set, is_heated):

    new_set_unheated = new_set[:, 0]
    new_set = new_set[:, 1]

    new_set_deriv = np.array(get_derivative(x_full, new_set))
    new_set_double_deriv = np.array(get_derivative(x_full, new_set_deriv))


    #for item in range(len(new_set)):
    #    print new_set[item], new_set_deriv[item], new_set_double_deriv[item], item
    #    if item == 300: break

    start_optim = np.argmin(np.abs(new_set_deriv[0:500]))

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


    return list(optim_set)

def get_x_matched_force_set(x_full, new_set, is_heated):

    new_set_unheated = new_set[:, 0]
    new_set = new_set[:, 1]

    new_set_force = new_set[:, 1]

    #for item in range(len(new_set)):
    #    print new_set[item], new_set_deriv[item], new_set_double_deriv[item], item
    #    if item == 300: break

    start_optim = np.argmin(np.abs(new_set_deriv[0:500]))

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


def get_x_matched_set(x_full, match_set, new_set, is_heated):

    new_set_unheated = new_set[:, 0]
    new_set = new_set[:, 1]

    matching_bounds = [0, 200]
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


    if is_heated == True:
        optim_set = optim_set_heated
        #optim_set = list(np.array(optim_set) + 29.5 - optim_set[0])
    else:
        optim_set = optim_set_unheated

    return optim_set


