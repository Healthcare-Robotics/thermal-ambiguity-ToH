#!/usr/bin/python
import numpy as np
from scipy.stats import norm
from cv2 import resize
import matplotlib
matplotlib.use( 'tkagg' )

from matplotlib import pyplot as plt


def plot_all_human_data_hist():
    s = open('study_data.txt', 'r').read()
    study_data_dict = eval(s)

    for item in study_data_dict:
        print(item)
        print('main study', item, len(study_data_dict[item]))

    x11 = np.array(study_data_dict['M|M_and_M|CW']).flatten()
    x10 = np.array(study_data_dict['M|M_and_W|CW']).flatten()
    x01 = np.array(study_data_dict['W|M_and_M|CW']).flatten()
    x00 = np.array(study_data_dict['W|M_and_W|CW']).flatten()
    temp_discrepancy = np.array(study_data_dict['temp_discrepancy']).flatten()

                                            
    print(len(temp_discrepancy), 'temp discrep')


    # create 3 data sets with 1,000 samples
    x_hist11 = []
    x_hist10 = []
    x_hist01 = []
    x_hist00 = []
    for i in range(0, len(temp_discrepancy)):
        if x11[i] == 1:
            x_hist11.append(temp_discrepancy[i])
        elif x10[i] == 1:
            x_hist10.append(temp_discrepancy[i])
        elif x01[i] == 1:
            x_hist01.append(temp_discrepancy[i])
        elif x00[i] == 1:
            x_hist00.append(temp_discrepancy[i])



    #Stack the data
    fig = plt.figure()
    ax = fig.gca()
    ax.set_yticklabels([0, 2, 4, 6, 8, 10, 12, 14, 16], fontsize=14)
    ax.set_xticklabels([-9,-8, -6, -4, -2, 0, 2, 4, 6, 8], fontsize=14)
    #ax.set_xticks()

    plt.xlabel('Finger Temperature Deviation', fontsize=14)
    plt.ylabel('Number of Trials', fontsize=14)
    n, bins, patches = plt.hist([np.array(x_hist11),np.array(x_hist10),np.array(x_hist01),np.array(x_hist00)], 32,
                                stacked=True, ec='k', range=(-8,8),
                                color= ['#7fc97f', '#fdc086', '#beaed4', '#ffff99'])

    # create legend
    plt.legend(['Guessed M is M, CW is M','Guessed M is M, CW is W','Guessed M is W, CW is M','Guessed M is W, CW is W'])

    ax.axvline(x=-3.5, color='r', linestyle='dashed', linewidth=2)
    ax.axvline(x=3.5, color='r', linestyle='dashed', linewidth=2)


    plt.show()


if __name__ == '__main__':

    plot_all_human_data_hist()

  
