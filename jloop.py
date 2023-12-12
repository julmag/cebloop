import numpy as np
import matplotlib.pyplot as plt

import water_tank as wt
from water_tank.Supplements import arm_data, video, visualize, pearson, mse_func


# Number of time steps
n = 16000
f_input, i_input, f_target, i_target, goals = arm_data(trial_duration=n)
input_stacked = np.concatenate((f_input, goals), axis=1)
target_stacked =  np.concatenate((f_target, i_target), axis=1)

# Parameters
N_in = 4 # number of inputs
N_out = 2 # number of outputs
N = 400 # number of neurons
g = 1.25 # scaling factor
tau = 3.3 # time constant
sparseness = 0.1 # sparseness of the recurrent weights
    
# Create network
inverse_net = wt.ESN(N, N_in, N_out, g, tau, sparseness)
forward_net = wt.ESN(N, N_in, N_out, g, tau, sparseness)


# Training / test
d_train = 8000
d_test = 8000

# Supervised training
inverse_net.loop_train(input_stacked[:d_train], target_stacked[:d_train], forward_net)

# Autoregressive test
inverse_net.loop_test(input_stacked[d_test:], target_stacked[d_test:], forward_net)

inverse_data = inverse_net.recorder.get()
forward_data = forward_net.recorder.get()


inverse_mse = mse_func(inverse_data, i_target, d_train)
forward_mse = mse_func(forward_data, f_target, d_train)
pearson(inverse_data, i_target)
pearson(forward_data, f_target)

video(forward_data['readout'][d_test:, :], f_target[d_test:,:])


visualize(inverse_data, i_target, d_train, "Inverse", d_train)
visualize(forward_data, f_target, d_train, "Forward", d_train)