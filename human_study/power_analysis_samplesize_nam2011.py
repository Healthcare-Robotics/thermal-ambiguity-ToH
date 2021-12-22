#!/usr/bin/python
import numpy as np
from scipy.stats import norm

N_per_cluster = 2  # number of examples per cluster. i.e. the average number of within-bounds trials per person.
#choose 2 by assuming that people's fingers will be in range roughly 2/3 of the time.

del_0 = -0.2
del_1 = 0.0
alpha = 0.05

if __name__ == '__main__':
  #threshold_numbers = [3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0]
  threshold_numbers = [3.5]
  K_calc_agg = []
  for threshold in threshold_numbers:

    s = open('pilot_data.txt', 'r').read()
    pilot_data_dict = eval(s)
    x_11k_u = np.array(pilot_data_dict['M|M_and_M|CW'])
    x_10k_u = np.array(pilot_data_dict['M|M_and_W|CW'])
    x_01k_u = np.array(pilot_data_dict['W|M_and_M|CW'])
    x_00k_u = np.array(pilot_data_dict['W|M_and_W|CW'])
    temp_discrepancy = np.array(pilot_data_dict['temp_discrepancy'])

    per_trial = True

    x_11k, x_10k, x_01k, x_00k = [], [], [], []
    last_i = 0
    for i in range(0,x_11k_u.shape[0]):
      switch = 0
      if per_trial == False:
        if np.abs(temp_discrepancy[i,0]) <= threshold and np.abs(temp_discrepancy[i,1]) <= threshold and np.abs(temp_discrepancy[i,2]) <= threshold:
          print(i)
          x_11k.append(x_11k_u[i,0]+x_11k_u[i,1]+x_11k_u[i,2])
          x_10k.append(x_10k_u[i,0]+x_10k_u[i,1]+x_10k_u[i,2])
          x_01k.append(x_01k_u[i,0]+x_01k_u[i,1]+x_01k_u[i,2])
          x_00k.append(x_00k_u[i,0]+x_00k_u[i,1]+x_00k_u[i,2])


      elif per_trial == True:
        for j in range(0,x_11k_u.shape[1]):
          if np.abs(temp_discrepancy[i,j]) <= threshold:
            if len(x_11k) <= i and switch == 0:
               switch = 1
               x_11k.append(x_11k_u[i,j])
               x_10k.append(x_10k_u[i,j])
               x_01k.append(x_01k_u[i,j])
               x_00k.append(x_00k_u[i,j])
            else:
               x_11k[-1] = x_11k[-1] + x_11k_u[i,j]
               x_10k[-1] = x_10k[-1] + x_10k_u[i,j]
               x_01k[-1] = x_01k[-1] + x_01k_u[i,j]
               x_00k[-1] = x_00k[-1] + x_00k_u[i,j]



    x_11k, x_10k, x_01k, x_00k = np.array(x_11k), np.array(x_10k), np.array(x_01k), np.array(x_00k)

    n_k = np.sum((x_11k, x_10k, x_01k, x_00k), axis = 0)*1.

    p_10k = x_10k/n_k
    p_01k = x_01k/n_k


    x_1ok = x_11k + x_10k*1.
    x_o1k = x_01k + x_11k*1.

    x_1oo = np.sum(x_1ok, axis = 0)*1.
    x_o1o = np.sum(x_o1k, axis = 0)*1.

    print(x_1oo, x_o1o)

    n_o = np.sum(n_k, axis = 0)

    print(n_o, 'sum')


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


    sig_b0 = np.sqrt(np.sum(np.square(p_10k - p_01k - del_0))/(K - 1))
    sig_b1 = np.sqrt(np.sum(np.square(p_10k - p_01k + del_0))/(K - 1))
    sig_w = np.sqrt(np.sum((p_10k + p_01k - np.square(p_10k - p_01k)))/(K - 1))


    print(sig_b0, 'sig_b0')
    print(sig_b1, 'sig_b1')
    print(sig_w, 'sig_w')


    z_1malph = 1.645 #confidence of 0.05
    print (z_1malph, n, K)

    u = (np.sqrt(np.square(sig_b0)+np.square(sig_w)/n)*z_1malph - np.sqrt(K)*(del_1-del_0))/np.sqrt(np.square(sig_b1)+np.square(sig_w)/n)
    print (u, 'u')

    v_0 = (np.square(sig_b0)+np.square(sig_w)/n)/K
    v_1 = (np.square(sig_b1)+np.square(sig_w)/n)/K
    u = (np.sqrt(v_0)*z_1malph - (del_1-del_0))/np.sqrt(v_1)
    print (u, 'triple calc u')

    #u = -.788 #we get this with del0 of -0.1 and del1 of 0.1
    print (1-norm.cdf(u), 'power')


    print( (sig_b0*z_1malph - np.sqrt(K)*(del_1 - del_0))/sig_b1, 'u as n goes to inf')

    #print ('Now we calculate the required K:')
    z_1mbeta = 0.8419 #power or beta of 0.8


    K_calc = np.square(np.sqrt(np.square(sig_b0)+np.square(sig_w)/N_per_cluster)*z_1malph + np.sqrt(np.square(sig_b1)+np.square(sig_w)/N_per_cluster)*z_1mbeta)/np.square(del_1-del_0)
    #print(K_calc, 'K calculated')

    K_calc_agg.append(K_calc)



    K_r = np.square(1.645 - 0.842)*(0.318 + sig_w/3.)/np.square(del_1-del_0)
    print (K_r)

  #print (threshold_numbers)
  print ("Number of subjects needed:", K_calc_agg)


  
