# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 17:14:24 2019

@author: Owen
"""
import matplotlib.pyplot as plt
import numpy as np

plt.figure(); plt.scatter(0, 0, c='r'); plt.grid()
plt.xlim(xmin=-50); plt.xlim(xmax=50)
plt.ylim(ymin=-50); plt.ylim(ymax=50)

n_samples = 300

actions = np.zeros((n_samples, 2))
states = np.zeros((n_samples, 2))
for i in range(n_samples):
    
    ###### actions ######
    t = 10
    v_init = np.array([0,0])
    a_init = np.array([np.random.uniform(-1,1), np.random.uniform(-1,1)])
    
    ###### states ######
    x_init = np.zeros((2))
    x_final = x_init + v_init*t + (1/2)*a_init*t**2
    print(i, a_init, x_final)

    plt.scatter(x_final[0], x_final[1], c='b')

    actions[i,:] = a_init
    states[i,:] = x_final

np.save('actions.npy', actions)
np.save('states.npy', states)



