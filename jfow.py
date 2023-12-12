import numpy as np
import matplotlib.pyplot as plt

import water_tank as wt
from water_tank.Supplements import arm_data, video, video2, video3, video4



# # Mackey-Glass chaotic time series
# def generate_mackey_glass(n, tau=17, beta=0.2, gamma=0.1, delta_t=0.1):
#     x = np.zeros(n)
#     x[0] = 0.9  # Initial condition

#     for t in range(1, n):
#         x_tau = x[max(0, t - tau)]
#         dxdt = beta * x_tau / (1 + x_tau**10) - gamma * x[t]
#         x[t] = x[t - 1] + delta_t * dxdt

#     return x

# Number of time steps
n = 6000
f_input, i_input, f_target, i_target, goals = arm_data(trial_duration=n)

# # Generate the Mackey-Glass time series
# mg = generate_mackey_glass(n)

# mg = (2.0 * (mg - mg.min()) / (mg.max() - mg.min()) - 1.0)[:,None]

# Parameters
N_in = 4 # number of inputs
N_out = 2 # number of outputs
N = 400 # number of neurons
g = 1.25 # scaling factor
tau = 3.3 # time constant
sparseness = 0.1 # sparseness of the recurrent weights
    
# Create network
net = wt.ESN(N, N_in, N_out, g, tau, sparseness)

# Training / test
d_train = 3000
d_test = 3000

# Supervised training
net.train(f_input[:d_train], f_target[:d_train])

# Autoregressive test
net.test_autoregress(f_input[d_test:], f_target[d_test:])
data = net.recorder.get()

mse = np.round((np.mean((f_target[d_test:,:] - data['readout'][d_test:, :]), axis=0)**2),4)
suptitle = "MSE X: " + str(mse[0]) + "; MSE Y: " + str(mse[1])
# Visualize
plt.figure(figsize=(12, 7))
plt.title("Autoregression")
plt.subplot(211)
plt.plot(f_target[:d_train+d_test,:], label='ground truth')
plt.plot(data['readout'][:d_train, :], label='prediction (training)')
plt.plot(np.linspace(d_train, d_train+d_test, d_test), data['readout'][d_train:, :], label='prediction (test)')
plt.legend()
plt.subplot(212)
plt.title(suptitle)
plt.plot(f_target[:d_train+d_test,:] - data['readout'][:, :], label='error')
plt.legend()
plt.show()

#video(data['readout'][:, :])
#video2(data['readout'][:, :], f_target[:,:])
#video3(data['readout'][:, :])
video4(data['readout'][d_test:, :], f_target[d_test:,:])
print("pogo")


