import numpy as np
import matplotlib.pyplot as plt

import water_tank as wt

# Mackey-Glass chaotic time series
def generate_mackey_glass(n, tau=17, beta=0.2, gamma=0.1, delta_t=0.1):
    x = np.zeros(n)
    x[0] = 0.9  # Initial condition

    for t in range(1, n):
        x_tau = x[max(0, t - tau)]
        dxdt = beta * x_tau / (1 + x_tau**10) - gamma * x[t]
        x[t] = x[t - 1] + delta_t * dxdt

    return x

# Number of time steps
n = 2000

# Generate the Mackey-Glass time series
mg = generate_mackey_glass(n)

mg = (2.0 * (mg - mg.min()) / (mg.max() - mg.min()) - 1.0)[:,None]

# Parameters
N_in = 1 # number of inputs
N_out = 1 # number of outputs
N = 400 # number of neurons
g = 1.25 # scaling factor
tau = 3.3 # time constant
sparseness = 0.1 # sparseness of the recurrent weights
    
# Create network
net = wt.ESN(N, N_in, N_out, g, tau, sparseness)

# Training / test
d_train = 500
d_test = 1000

# Supervised training
net.train(mg[:d_train], mg[1:d_train+1])

# Autoregressive test
net.autoregressive(duration=d_test)
data = net.recorder.get()

# Visualize
plt.figure(figsize=(12, 7))
plt.title("Autoregression")
plt.subplot(211)
plt.plot(mg[1:d_train+d_test+1, 0], label='ground truth')
plt.plot(data['readout'][:d_train, 0], label='prediction (training)')
plt.plot(np.linspace(d_train, d_train+d_test, d_test), data['readout'][d_train:, 0], label='prediction (test)')
plt.legend()
plt.subplot(212)
plt.plot(mg[1:d_train+d_test+1, 0] - data['readout'][:, 0], label='error')
plt.legend()
plt.show()



