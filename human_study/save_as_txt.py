#!/usr/bin/python
import numpy as np
from scipy.stats import norm


if __name__ == '__main__':

  data_dict = {}

  data_dict['subject_num'] = ['A27', 'A35', 'A43', 'A40', 'A31',
                              'A23', 'A02', 'A34', 'A05', 'A29',
                              'A48', 'A09', 'A13']
  data_dict['M|M_and_M|CW'] = [[1,1,1],[1,0,0],[0,1,1],[1,0,0],[0,1,0],
                               [0,0,0],[1,0,0],[1,1,1],[1,1,1],[0,0,1],
                               [1,1,1],[1,0,0],[1,0,1]]
  data_dict['M|M_and_W|CW'] = [[0,0,0],[0,0,0],[1,0,0],[0,0,0],[0,0,0],
                               [1,0,1],[0,0,0],[0,0,0],[0,0,0],[0,0,0],
                               [0,0,0],[0,1,1],[0,0,0]]
  data_dict['W|M_and_M|CW'] = [[0,0,0],[0,1,1],[0,0,0],[0,1,1],[1,0,1],
                               [0,1,0],[0,1,1],[0,0,0],[0,0,0],[1,1,0],
                               [0,0,0],[0,0,0],[0,1,0]]
  data_dict['W|M_and_W|CW'] = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],
                               [0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],
                               [0,0,0],[0,0,0],[0,0,0]]
  data_dict['temp_discrepancy'] = [[6.0,6.5,5.5],[4.0,5.5,4.0],[-5.0,2.0,3.5],[6.0,6.5,6.5],[5.0,5.0,5.0],
                                   [-5.0,0.5,-0.5],[2.0,4.0,6.0],[0.5,0.0,1.0],[5.5,5.0,2.5],[6.0,5.5,2.5],
                                   [2.0,3.5,3.0],[-2.0,-3.0,-5.0],[3.5,3.5,1.0]]
  data_dict['ethnicity'] = []
  data_dict['gender'] = []
  data_dict['age'] = []

  for item in data_dict:
    print('pilot study', item, len(data_dict[item]))

  with open('pilot_data.txt', 'w') as f:
    print(data_dict, file=f)




  study_data_dict = {}

  study_data_dict['subject_num'] = ['S98', 'S06', 'S93', 'S78', 'S33',
                                    'S19', 'S21', 'S26', 'S54', 'S18',
                                    'S72', 'S79', 'S89', 'S77', 'S63',
                                    'S90', 'S37', 'S53', 'S73', 'S34',
                                    'S61', 'S11', 'S14', 'S30', 'S75',
                                    'S27', 'S67', 'S83', 'S86', 'S95',
                                    'S58', 'S97']
  study_data_dict['M|M_and_M|CW'] = [[1,1,1],[1,0,1],[1,1,0],[0,0,1],[1,1,1],
                                     [1,1,1],[1,0,1],[1,1,1],[0,1,1],[1,1,1],
                                     [1,1,1],[1,1,1],[1,1,1],[1,1,1],[0,0,1],
                                     [1,1,1],[1,1,1],[1,1,1],[1,1,0],[1,1,0],
                                     [1,1,1],[1,0,1],[1,1,1],[0,1,1],[1,1,1],
                                     [1,0,1],[1,0,1],[1,1,1],[1,1,1],[0,1,1],
                                     [1,1,1],[0,1,0]]
  study_data_dict['M|M_and_W|CW'] = [[0,0,0],[0,1,0],[0,0,0],[0,0,0],[0,0,0],
                                     [0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],
                                     [0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],
                                     [0,0,0],[0,0,0],[0,0,0],[0,0,1],[0,0,0],
                                     [0,0,0],[0,1,0],[0,0,0],[0,0,0],[0,0,0],
                                     [0,1,0],[0,1,0],[0,0,0],[0,0,0],[0,0,0],
                                     [0,0,0],[0,0,0]]
  study_data_dict['W|M_and_M|CW'] = [[0,0,0],[0,0,0],[0,0,1],[1,1,0],[0,0,0],
                                     [0,0,0],[0,1,0],[0,0,0],[1,0,0],[0,0,0],
                                     [0,0,0],[0,0,0],[0,0,0],[0,0,0],[1,0,0],
                                     [0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,1],
                                     [0,0,0],[0,0,0],[0,0,0],[1,0,0],[0,0,0],
                                     [0,0,0],[0,0,0],[0,0,0],[0,0,0],[1,0,0],
                                     [0,0,0],[1,0,1]]
  study_data_dict['W|M_and_W|CW'] = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],
                                     [0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],
                                     [0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,1,0],
                                     [0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],
                                     [0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],
                                     [0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],
                                     [0,0,0],[0,0,0]]
  study_data_dict['temp_discrepancy'] = [[-4.8, -5.0, -6.8], [-1.2, -2.0, -1.1], [-4.0, -4.4, -4.5], [-3.6, -3.7,  0.4], [-0.7, -0.8, -1.6],
                                        [ 0.8,  1.0,  1.3], [-0.4, -0.8, -1.0], [-0.7, -1.2, -2.5], [-4.2, -2.0, -3.0], [-1.1, -1.3, -1.0],
                                        [ 0.1, -0.1,  0.5], [-0.5,  0.6,  1.0], [ 3.3,  4.3,  3.9], [-0.9, -0.6, -0.7], [-3.0, -1.2, -2.0],
                                        [-0.7, -1.6, -1.5], [-1.6, -2.0, -0.7], [-1.5, -1.9, -2.0], [ 1.0, -0.6,  0.3], [-3.3, -0.7, -2.1],
                                        [-0.2,  0.1, -0.2], [ 2.5,  2.6,  1.8], [-0.7, -0.9,  3.4], [-7.2, -0.3,  2.5], [-0.1,  0.7,  1.9],
                                        [ 1.9,  3.0,  1.0], [ 2.9,  3.7,  2.3], [ 0.4,  2.3,  1.6], [ 0.8,  0.0,  0.2], [-4.4, -2.9, -3.2],
                                        [ 0.5,  2.0,  1.4], [-6.3, -3.0, -3.8]]
  study_data_dict['ethnicity'] = ['Asian', 'Mixed', 'Asian', 'Asian', 'Asian',
                                 'White', 'White', 'White', 'Asian', 'Asian',
                                 'Asian', 'Asian', 'White', 'White', 'White',
                                 'White', 'Asian', 'White', '-', 'Asian',
                                 'White', 'Native American + Hispanic', 'Asian', 'Asian', 'Asian',
                                 'Asian', 'Asian', 'Asian', '-', 'White',
                                 'Asian+White', 'Asian']
  study_data_dict['gender'] = ['F', 'M', 'F', 'F', 'F',
                              'M', 'F', 'F', 'F', 'M',
                              'M', 'M', 'M', 'M', 'F',
                              'F', 'F', 'M', 'M', 'M',
                              'M', 'M', 'F', 'F', 'M',
                              'M', 'F', 'F', '-', 'F',
                              'M', 'M']
  study_data_dict['age'] = [18, 19, 20, 19, 23,
                           20, 22, 20, 18, 19,
                           18, 20, 23, 26, 22,
                           22, 21, 27, 20, 27,
                           38, 24, 27, 22, 23,
                           23, 25, 18, None, 19,
                           20, 24]

  for item in study_data_dict:
    print('main study', item, len(study_data_dict[item]))



  with open('study_data.txt', 'w') as f:
    print(study_data_dict, file=f)