# @title Supplements
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.animation import FuncAnimation
import scipy
from sklearn.model_selection import train_test_split
import pandas as pd
import os



class PlanarArm (object):
    """
    Forward model of a 2D arm.
    """
    def __init__(self,
            shoulder=np.array([0.0, 0.0]),
            # arm_lengths = np.array([0.5, 0.5])
            arm_lengths = np.array([1.8, 1.8]),

        ) -> None:
        """
        Attributes:
        * shoulder: position of the shoulder (default: (0, 0))
        * arm_lengths: length of each segment (default: (0.5, 0.5))
        """
        self.shoulder = shoulder # Shoulder position
        self.arm_lengths = arm_lengths # Length of each segment

        self.reset()



    def reset(self):

        # Current position
        self.angles = self.sample_angles()
        self.elbow, self.hand = self.forward(self.angles)

        # Target to track
        self.target = self.sample_angles()
        self.smoothed_target = self.target.copy()


    def forward(self, angles):
        """
        Hand position as a function of the shoulder and elbow angles:

        Attributes:
        * angles: [theta_shoulder, theta_elbow] in radians

        Returns:
        * elbow: [x_elbow, y_elbow]
        * hand: [x_hand, y_hand]
        """

        elbow = np.array(
            [
                self.shoulder[0] + self.arm_lengths[0] * np.cos(angles[0]),
                self.shoulder[1] + self.arm_lengths[0] * np.sin(angles[0]),
            ]
        )
        hand = np.array(
            [
                elbow[0] + self.arm_lengths[1] * np.cos(angles[0] + angles[1]),
                elbow[1] + self.arm_lengths[1] * np.sin(angles[0] + angles[1]),
            ]
        )

        return elbow, hand


    def inverse(self, hand):
        """
        Inverse kinematics model.

        Source: <https://robotacademy.net.au/lesson/inverse-kinematics-for-a-2-joint-robot-arm-using-geometry/>

        Attributes:
        * hand: [x_hand, y_hand]

        Returns:
        * angles: [theta_shoulder, theta_elbow] in radians
        * hand: [x_hand, y_hand]
        * elbow: [x_elbow, y_elbow]
        """

        x, y = hand[0], hand[1]
        l1, l2 = self.arm_lengths[0], self.arm_lengths[1]
        hypothenuse = x**2 + y**2

        # if hypothenuse >1 and hypothenuse <1.1:
        #     hypothenuse = 1.0

        # Compute angles using inverse kinematics

        theta_elbow = np.arccos(
            (hypothenuse - l1**2 - l2**2) / (2 * l1 * l2)
        )
        try:
            theta_shoulder = np.arctan(y / x) - np.arctan((l2 * np.sin(theta_elbow))/(l1 + l2 * np.cos(theta_elbow)))
        except:
            print ('X: ', x, '   Y: ', y,'   theta_elbow: ', theta_elbow, '  l1 + l2:', l1, l2 )

        # Avoid breaking the elbow
        if theta_shoulder < 0.0:
            theta_shoulder += np.pi



        angles = np.array([theta_shoulder, theta_elbow])

        elbow = np.array([l1 * np.cos(theta_shoulder), l2 * np.sin(theta_shoulder)])



        return elbow, hand, angles

    def sample_angles(self):
        "Returns sampled angles between 0 and pi."
        # return np.random.uniform(0., np.pi, 2)

        l1, l2 = self.arm_lengths[0], self.arm_lengths[1]

        # x,y = np.random.uniform(-(self.arm_lengths[0]), 0.), np.random.uniform(0, (self.arm_lengths[0]))
        x,y = np.random.choice((np.random.uniform(-1.0,-0.7),np.random.uniform(0.7,1.2))), np.random.choice((np.random.uniform(-1,-0.7),np.random.uniform(1,1.5)))

        # Compute angles using inverse kinematics
        theta_elbow = np.arccos(((x**2 + y**2) - l1**2 - l2**2) / (2 * l1 * l2))
        theta_shoulder = np.arctan(y / x) - np.arctan((l2 * np.sin(theta_elbow))/(l1 + l2 * np.cos(theta_elbow)))
        theta_shoulder = theta_shoulder * (-1)

        angles = np.array([theta_shoulder, theta_elbow])



        while np.any(np.isnan(angles)) == True:

            x,y = np.random.choice((np.random.uniform(-1.0,-0.7),np.random.uniform(0.7,1.2))), np.random.choice((np.random.uniform(-1,-0.7),np.random.uniform(1,1.5)))

            theta_elbow = np.arccos(((x**2 + y**2) - l1**2 - l2**2) / (2 * l1 * l2))
            theta_shoulder = np.arctan(y / x) - np.arctan((l2 * np.sin(theta_elbow))/(l1 + l2 * np.cos(theta_elbow)))
            theta_shoulder = theta_shoulder * (-1)

            angles = np.array([theta_shoulder, theta_elbow])


        return angles


    def random_trajectory(self, duration, speed=0.05, proba_switch=0.0, switch_target=True):

        # Record
        elbow_positions = [self.elbow]
        hand_positions = [self.hand]
        angles = [self.angles]
        delta_angles = [np.array([0.0, 0.0])]
        goal = [self.target]

        temp_target = self.sample_angles()

        while (((temp_target[0] - self.target[0])**2 + (temp_target[1] - self.target[1])**2)**0.5) < 0.5 :
            temp_target = self.sample_angles()

        self.target = temp_target



        for t in range(int(duration)):

            if switch_target==True:
                # Target updating
                if np.random.random() < proba_switch:
                    temp_target = self.sample_angles()


                    while (((temp_target[0] - self.target[0])**2 + (temp_target[1] - self.target[1])**2)**0.5) < 0.5 :
                        temp_target = self.sample_angles()

                    self.target = temp_target


                        # self.target = self.sample_angles()

            # Smooth the target
            self.smoothed_target = (1.- speed) * self.smoothed_target + speed * self.target

            # Momentum to smooth the movements
            # new_angle = (1.- speed) * self.angles + speed * self.smoothed_target
            new_angle = (1.- speed) * self.angles + speed * self.target


            # if np.sum(abs(self.angles-self.target)) >0.01:
            #     if self.angles < 0 and self.angles < self.target:
            #         new_angle = self.angles + (speed * self.target)*(-1)
            #     else:
            #         new_angle = self.angles + speed * self.target




            delta = new_angle - self.angles

            self.angles = new_angle

            # Forward model
            self.elbow, self.hand = self.forward(self.angles)

            # Track values
            elbow_positions.append(self.elbow)
            hand_positions.append(self.hand)
            angles.append(self.angles)
            delta_angles.append(delta)
            goal.append(self.target)

        goal = np.array(goal)
        elbow_positions = np.array(elbow_positions)
        hand_positions = np.array(hand_positions)
        angles = np.array(angles)
        delta_angles = np.array(delta_angles)

        return elbow_positions, hand_positions, angles, delta_angles, goal

    def intersect (self, prediction):

        b = (prediction[0]**2 + prediction[1]**2)**0.5
        a = prediction[0]
        c = prediction[1]

        alpha = math.acos(((b**2 + c **2 - a **2) / (2 * b * c)))
        beta = math.radians(90)
        gamma = math.radians(180 - (math.degrees(beta) + math.degrees(alpha)))


        intersect_b = np.sum(self.arm_lengths)
        intersect_x = (intersect_b / math.sin(beta))*math.sin(alpha)
        intersect_y = (intersect_b / math.sin(beta))*math.sin(gamma)

        return np.array([intersect_x, intersect_y])





def arm_data (trial_duration, test_size=2000, forward=True):
    #Generating Input and Target Data
    arm = PlanarArm() # Arm Class of a 2D Arm
    elbow_positions, hand_positions, angles, delta_angles, goals = arm.random_trajectory(duration=trial_duration+10, speed=0.01, proba_switch = 0.05)

    hand_input = hand_positions[:trial_duration] # x,y position of hand 
    hand_target = hand_positions[1:trial_duration+1] # desired x,y position of hand?

    d_angles = delta_angles[1:trial_duration+1]
    d_angles_target = delta_angles[2:trial_duration+2]

    goals = goals[2:trial_duration+2] # target angle-differences of arm and shoulder?

    f_target = hand_target[:,:]
    f_input = np.concatenate((hand_input, d_angles), axis=1)

    i_target = d_angles_target[:,:]
    i_input = np.concatenate((hand_input, goals),axis=1)


    # pos_train, pos_test, pos_target_train, pos_target_test = train_test_split(hand_input, hand_target, test_size=test_size, shuffle=False)
    # djoint_train, djoint_test, djoint_target_train, djoint_target_test = train_test_split(d_angles, d_angles_target, test_size=test_size, shuffle=False)
    # goals_train = goals[:-test_size,:]
    # goals_test = goals[-test_size:,:]

#     return pos_train, pos_test, pos_target_train, pos_target_test, djoint_train, djoint_test, djoint_target_train, djoint_target_test, goals_train, goals_test

# pos_train, pos_test, pos_target_train, pos_target_test, djoint_train, djoint_test, djoint_target_train, djoint_target_test, goals_train, goals_test = arm_data(trial_duration=20000, test_size=4000)

    return f_input[:trial_duration,:], i_input[:trial_duration,:], f_target[:trial_duration,:], i_target[:trial_duration,:], goals[:trial_duration,:]



def video(coordinates_over_time_1, coordinates_over_time_2):
    import pygame
    import numpy as np

    # Initialize Pygame
    pygame.init()

    # Constants
    SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
    POINT_RADIUS = 3
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    BLUE = (0, 0, 255)
    RED = (255, 0, 0)

    # Find the overall range of coordinates for both sets
    min_x = min(np.min(coordinates_over_time_1[:, 0]), np.min(coordinates_over_time_2[:, 0]))
    min_y = min(np.min(coordinates_over_time_1[:, 1]), np.min(coordinates_over_time_2[:, 1]))
    max_x = max(np.max(coordinates_over_time_1[:, 0]), np.max(coordinates_over_time_2[:, 0]))
    max_y = max(np.max(coordinates_over_time_1[:, 1]), np.max(coordinates_over_time_2[:, 1]))
    range_x = max_x - min_x
    range_y = max_y - min_y

    # Scale the coordinates to fit the screen
    def scale_coordinates(coord):
        x, y = coord
        scaled_x = int(((x - min_x) / range_x) * SCREEN_WIDTH)
        scaled_y = int(((y - min_y) / range_y) * SCREEN_HEIGHT)
        return scaled_x, scaled_y

    scaled_coordinates_1 = np.array([scale_coordinates(coord) for coord in coordinates_over_time_1])
    scaled_coordinates_2 = np.array([scale_coordinates(coord) for coord in coordinates_over_time_2])

    # Set up the screen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    running = True
    frame = 0

    while running and frame < min(len(scaled_coordinates_1), len(scaled_coordinates_2)):
        screen.fill(WHITE)
        
        # Draw points for the first set of coordinates for the current frame
        if frame < len(scaled_coordinates_1):
            current_point_1 = scaled_coordinates_1[frame]
            pygame.draw.circle(screen, BLUE, current_point_1, POINT_RADIUS)
        
        # Draw points for the second set of coordinates for the current frame
        if frame < len(scaled_coordinates_2):
            current_point_2 = scaled_coordinates_2[frame]
            pygame.draw.circle(screen, RED, current_point_2, POINT_RADIUS)
        
        pygame.display.flip()
        clock.tick(60)

        frame += 1

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()

def mse_func(data, target, split):
    data = data['readout']
    mse_values = np.mean((target[split:] - data[split:]), axis=0) ** 2
    rounded_mse = [f"{val:.3e}" if abs(val) < 1e-3 else round(val, 3) for val in mse_values]
    return [float(val) for val in rounded_mse]

def pearson(data, target):
    data = data['readout']
    if data.shape != target.shape:
        raise ValueError("Arrays must have the same shape")

    num_cols = data.shape[1]
    correlations = np.zeros(num_cols)

    for i in range(num_cols):
        correlation, _ = scipy.stats.pearsonr(data[:, i], target[:, i])
        correlations[i] = correlation

    return correlations


def visualize(data, target, split, name, start_row=0, end_row=None, save_pic=False):
    mse = mse_func(data, target, split) # important that it heappens before     data = data['readout'] because the mse_func does the same
    
    data = data['readout']
    
    if end_row is None:
        end_row = data.shape[0]
    
    
    name = str(name)
    error = np.abs(data - target)

    plt.figure(figsize=(12, 8))

    plt.subplot(211)
    plt.plot(data[start_row:end_row, 0], linewidth=0.5, label='Prediction (x)')
    plt.plot(target[start_row:end_row, 0], linewidth=1.5, alpha=0.5, label='Target (x)')
    if start_row <= split <= end_row:
        plt.axvline(x=split-start_row, color='red', linestyle='--', label='Split Point')


    plt.title(name + ' Model Prediction vs. Target (x)' + f" MSE: {mse[0]}")
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)

    plt.subplot(212)
    plt.plot(data[start_row:end_row, 1], linewidth=0.5, label='Prediction (y)')
    plt.plot(target[start_row:end_row, 1], linewidth=1.5, alpha=0.5, label='Target (y)')
    if start_row <= split <= end_row:
        plt.axvline(x=split-start_row, color='red', linestyle='--', label='Split Point')

    plt.title(name + ' Model Prediction vs. Target (y)' + f" MSE: {mse[1]}")
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)

    plt.subplots_adjust(hspace=0.5)  

    if save_pic:
        directory = 'Pictures/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(directory + name + "_Model.png")
        plt.close() 
    
    
    plt.figure(figsize=(12, 8))

    plt.subplot(211)
    plt.plot(error[start_row:end_row, 0], linewidth=0.5, label='Error (x)')
    if start_row <= split <= end_row:
        plt.axvline(x=split-start_row, color='red', linestyle='--', label='Split Point')

    plt.title(name + ' Absolute Error (x)' + f" MSE: {mse[0]}")
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)

    plt.subplot(212)
    plt.plot(error[start_row:end_row, 1], linewidth=0.5, label='Error (y)')
    if start_row <= split <= end_row:
        plt.axvline(x=split-start_row, color='red', linestyle='--', label='Split Point')

    plt.title(name + ' Absolute Error (y)' + f" MSE: {mse[1]}")
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)

    plt.subplots_adjust(hspace=0.5)  

    if save_pic:
        directory = 'Pictures/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(directory + name + "_Model_Error.png")
        plt.close() 

def vis_sum_2p(data, name, save_pic=False):
    avg = np.mean(data, axis=0)
    rounded_avg = [f"{val:.3e}" if abs(val) < 1e-3 else round(val, 3) for val in avg]
    
    below_average = (data < rounded_avg).sum(axis=0)  
    above_average = (data > rounded_avg).sum(axis=0)  
    
    plt.figure(figsize=(12, 8))
    plt.suptitle(name)

    plt.subplot(211)
    plt.plot(data[:, 0], label="x", color="red")
    plt.title(f" Average X: {rounded_avg[0]}, Above/below Avg.: {above_average[0]}/{below_average[0]}")
    plt.grid(True)
    
    plt.subplot(212)
    plt.plot(data[:, 1], label="y", color="blue")
    plt.title(f" Average Y: {rounded_avg[1]}, Above/below Avg.: {above_average[1]}/{below_average[1]}")
    plt.grid(True)

    plt.subplots_adjust(hspace=0.5)  
    
    if save_pic:
        directory = 'Pictures/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(directory + name + "_Summary_2p.png")
        plt.close() 



def vis_sum_4p(mse, pearson, name, save_pic=False):
    # Calculate average values for MSE and Pearson metrics
    mse_avg = np.mean(mse, axis=0)
    mse_rounded_avg = [f"{val:.3e}" if abs(val) < 1e-3 else round(val, 3) for val in mse_avg]

    pearson_avg = np.mean(pearson, axis=0)
    pearson_rounded_avg = [f"{val:.3e}" if abs(val) < 1e-3 else round(val, 3) for val in pearson_avg]

    # Calculate the number of values below and above the respective averages for MSE and Pearson metrics
    mse_below_average = (mse < mse_avg).sum(axis=0)  
    mse_above_average = (mse > mse_avg).sum(axis=0)  
    
    pearson_below_average = (pearson < pearson_avg).sum(axis=0)  
    pearson_above_average = (pearson > pearson_avg).sum(axis=0)  

    median_mse = np.median(mse,axis=0)
    median_deviation_mse = np.abs(mse-median_mse)
    mad_mse = np.median(median_deviation_mse,axis=0)

    median_pearson = np.median(pearson,axis=0)
    median_deviation_pearson = np.abs(pearson-median_pearson)
    mad_pearson = np.median(median_deviation_pearson,axis=0)
        
    # Create a new figure for plotting
    plt.figure(figsize=(12, 12))
    plt.suptitle(name)  # Set the title for the entire figure

    # Subplot for MSE X values
    plt.subplot(221)
    plt.plot(mse[:, 0], label="X", color="red")
    plt.title(f"MSE for X\nAverage: {mse_rounded_avg[0]}, \nMAD: {mad_mse[0]}, \nAbove/Below Avg.: {mse_above_average[0]}/{mse_below_average[0]}")
    plt.xlabel("Data Points")
    plt.ylabel("MSE Value")
    plt.grid(True)
    
    # Subplot for MSE Y values
    plt.subplot(222)
    plt.plot(mse[:, 1], label="Y", color="blue")
    plt.title(f"MSE for Y\nAverage: {mse_rounded_avg[1]}, \nMAD: {mad_mse[1]},  \nAbove/Below Avg.: {mse_above_average[1]}/{mse_below_average[1]}")
    plt.xlabel("Data Points")
    plt.ylabel("MSE Value")
    plt.grid(True)

    # Subplot for Pearson R² X values
    plt.subplot(223)
    plt.plot(pearson[:, 0], label="X", color="red")
    plt.title(f"Pearson R² for X\nAverage: {pearson_rounded_avg[0]},\nMAD: {mad_pearson[0]}, \nAbove/Below Avg.: {pearson_above_average[0]}/{pearson_below_average[0]}")
    plt.xlabel("Data Points")
    plt.ylabel("Pearson R² Value")
    plt.grid(True)

    # Subplot for Pearson R² Y values
    plt.subplot(224)
    plt.plot(pearson[:, 1], label="Y", color="blue")
    plt.title(f"Pearson R² for Y\nAverage: {pearson_rounded_avg[1]},\nMAD: {mad_pearson[1]}, \nAbove/Below Avg.: {pearson_above_average[1]}/{pearson_below_average[1]}")
    plt.xlabel("Data Points")
    plt.ylabel("Pearson R² Value")
    plt.grid(True)
    
    # Adjust the spacing between subplots
    plt.subplots_adjust(hspace=0.5, wspace=0.3)  
    
    # Save the plot as an image if save_pic is True
    if save_pic:
        directory = 'Pictures/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(directory + name + "_Summary_4p.png")
        plt.close() 



