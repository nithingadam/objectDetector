import math 
import numpy as np 
# E = math.e
# import nnfs
# nnfs.init()
outputs = [[4.8, 1.21, 2.385],
           [1.5,6.7,9.7],
           [2.3,4.1,9.8]]

exp_val = np.exp(outputs)
print(exp_val)

print(np.sum(outputs, axis=1, keepdims=True))

norm_val = exp_val / np.sum(exp_val, axis = 1, keepdims=True)

print(norm_val)
