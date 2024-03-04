import copy
import math
import random
import sys

import gym
import numpy as np
from gym import spaces
# from gym.envs.classic_control import rendering
from matplotlib.colors import hsv_to_rgb
import multiprocessing as mp

import alg_parameters
from map_generator import *
import torch
from astar_4 import manhattan_distance

from alg_parameters import *
from od_mstar3 import od_mstar
from od_mstar3.col_set_addition import NoSolutionError
#
# dic = {'(1, 2)': [(1, 5)], '(1, 5)': [(1, 2), (1, 9)], '(1, 9)': [(1, 5), (1, 18)], '(1, 18)': [(1, 9)]}
#
# ims = dic.items()
# # print(list(ims))
# value = dic.values()
# list_val = list(value)
# for i in range(len(list_val)):
#     for j in range(len(list_val[i])):
#         while len(list_val[i]) < 4:
#             list_val[i].append((0, 0))
# array_val = np.array(list_val)
# print(array_val)
# print(array_val.shape)
#
# list_kys = list(dic.keys())
# # for i in range(len(list_kys)):
# #     list_kys[i] = list(list_kys[i])
# print(list_kys)
# # val_pad = np.pad(list_val, ((0, 2), (0, 0), (0, 0)), 'constant')
# # print(val_pad)

# b = torch.Tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
# print(b.shape)
# a = torch.Tensor([[[1, 1, 1, 1, 1, 1],
#                    [1, 1, 1, 1, 1, 1],
#                    [1, 1, 1, 1, 1, 1],
#                    [1, 1, 1, 1, 1, 1],
#                    [1, 1, 1, 1, 1, 1],
#                    [1, 1, 1, 1, 1, 1],
#                    [1, 1, 1, 1, 1, 1],
#                    [1, 1, 1, 1, 1, 1],
#                    [1, 1, 1, 1, 1, 1],
#                    [1, 1, 1, 1, 1, 1]]])
# print(a.shape)
#
# c = b.repeat_interleave(a.size(2), dim=1)
# print(c)
#
# d = c.reshape(a.size(0), a.size(1), a.size(2))
# e = torch.mul(a, d)
# print(d)

import numpy as np

# Your rectangular 2D NumPy array
rectangular_array = np.array([
    [-1, -1, -1, -1, -1, -1],
    [4, 5 ,1 ,1, 1, 2],
    [7, 8, 1, 4, 1, 4]
])

# Determine the desired size for the square array
desired_size = max(rectangular_array.shape)

# Create a new square array filled with -1
square_array = np.full((desired_size, desired_size), 1)

# Copy the values from the original rectangular array to the center of the square array
row_offset = 0
col_offset = desired_size - rectangular_array.shape[1]

square_array[row_offset:row_offset+rectangular_array.shape[0], col_offset:col_offset+rectangular_array.shape[1]] = rectangular_array

# square_array is now your padded square array
print(square_array)

# import numpy as np
#
# # Your rectangular 2D NumPy array
# rectangular_array = np.array([
#     [-1, -1, -1, -1, -1],
#     [4, 5 ,1 ,1, 1],
#     [7, 8, 1, 4, 1]
# ])
#
# # Determine the desired size for the square array
# desired_size = max(rectangular_array.shape)
#
# # Create a new square array filled with -1
# square_array = np.full((desired_size, desired_size), 1)
#
# # Calculate the offsets needed to place the original data in the center of the square array
# row_offset = (desired_size - rectangular_array.shape[0]) // 2
# col_offset = (desired_size - rectangular_array.shape[1]) // 2
#
# # Calculate the row and column indices for copying the original data into the square array
# row_start = row_offset
# row_end = row_offset + rectangular_array.shape[0]
# col_start = col_offset
# col_end = col_offset + rectangular_array.shape[1]
#
# # Copy the values from the original rectangular array to the center of the square array
# square_array[row_start:row_end, col_start:col_end] = rectangular_array
#
# # square_array is now your padded square array with -1 values on the opposite sides
# print(square_array)
