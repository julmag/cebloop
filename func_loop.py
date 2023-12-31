import numpy as np
import matplotlib.pyplot as plt

import water_tank as wt
from water_tank.Supplements import arm_data, video, visualize, pearson, mse_func, vis_sum_2p, vis_sum_4p, timer
import time

def loop_model(version):
    # Number of time steps
    n = 50
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
    split = 10


    # Supervised training
    inverse_net.loop_train(input_stacked[:split], target_stacked[:split], forward_net)

    # Autoregressive test
    inverse_net.loop_test(input_stacked[split:], target_stacked[split:], forward_net)

    inverse_data = inverse_net.recorder.get()
    forward_data = forward_net.recorder.get()


    inverse_mse = mse_func(inverse_data, i_target, split)
    forward_mse = mse_func(forward_data, f_target, split)
    inverse_pearson = pearson(inverse_data, i_target)
    forward_pearson = pearson(forward_data, f_target)

    #video(forward_data['readout'][split:, :], f_target[split:,:])

    if version %10 == 0:
        visualize(inverse_data, i_target, split, "Inverse " + str(version+1), split, save_pic = True)
        visualize(forward_data, f_target, split, "Forward " + str(version+1), split, save_pic = True)
    
    return inverse_mse, forward_mse, inverse_pearson, forward_pearson




loops = 2
start_time = time.time()
im, ip, fm, fp = [], [], [], []

for i in range(loops):
    loop_start = time.time()

    # Assuming loop_model(i) is a placeholder function for your calculations
    inverse_mse, forward_mse, inverse_pearson, forward_pearson = loop_model(i)
    im.append(inverse_mse)
    ip.append(inverse_pearson)
    fm.append(forward_mse)
    fp.append(forward_pearson)

    loop_end = time.time()
    loop_time = loop_end - loop_start
    time_elapsed = loop_end - start_time

    formatted_loop_time, formatted_time_elapsed, formatted_remaining_time = timer(loop_time , time_elapsed , loops, i)

    print(f"Loop {i + 1}/{loops} | Time taken: {formatted_loop_time} | Elapsed: {formatted_time_elapsed} | Remaining: {formatted_remaining_time}")
    

time_elapsed = time.time() - start_time
_, formatted_time_elapsed, _ = timer(time_elapsed , time_elapsed , loops, loops)

print(f"Total execution time: {formatted_time_elapsed}")



inverse_mse = np.array(im)
inverse_pearson = np.array(ip)
forward_mse = np.array(fm)
forward_pearson = np.array(fp)

vis_sum_4p(inverse_mse, inverse_pearson, "Inverse", save_pic=True)
vis_sum_4p(forward_mse, forward_pearson, "Forward", save_pic=True)


# Printing the formatted time

