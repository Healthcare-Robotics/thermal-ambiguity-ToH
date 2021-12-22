#!/usr/bin/python
import numpy as np
from scipy.stats import norm

if __name__ == '__main__':
  threshold_numbers = [0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0] 
  within_region_all = []
  outside_region_all = []

  K_calc_agg = []
  for threshold in threshold_numbers:

    s = open('pilot_data.txt', 'r').read()
    pilot_data_dict = eval(s)
    x_11 = np.array(pilot_data_dict['M|M_and_M|CW'])
    x_10 = np.array(pilot_data_dict['M|M_and_W|CW'])
    x_01 = np.array(pilot_data_dict['W|M_and_M|CW'])
    x_00 = np.array(pilot_data_dict['W|M_and_W|CW'])
    temp_discrepancy = np.array(pilot_data_dict['temp_discrepancy'])


    within_region = 0
    outside_region = 0
    total_within_ideal = np.sum(x_11)
    total_outside_ideal = np.sum(x_10) + np.sum(x_01)

    per_trial = True
    last_i = 0
    for i in range(0,x_11.shape[0]):
      switch = 0
      if per_trial == False:
        if np.abs(temp_discrepancy[i,0]) <= threshold and np.abs(temp_discrepancy[i,1]) <= threshold and np.abs(temp_discrepancy[i,2]) <= threshold:
          pass

      elif per_trial == True:
        for j in range(0,x_11.shape[1]):
          if np.abs(temp_discrepancy[i,j]) <= threshold:
            within_region = within_region + x_11[i,j]
          else:
            outside_region = outside_region + x_10[i,j] + x_01[i,j]

    within_region_all.append(within_region)
    outside_region_all.append(outside_region)

  within_region_perc = np.array(within_region_all)*1./total_within_ideal
  outside_region_perc = np.array(outside_region_all)*1./total_outside_ideal



  print(within_region_all)
  print(outside_region_all)

  print(threshold_numbers)
  print(within_region_perc + outside_region_perc)

  max_idx = np.argmax(within_region_perc + outside_region_perc)

  opt_threshold = threshold_numbers[max_idx]
  print("the optimum threshold is: ", opt_threshold)


  
