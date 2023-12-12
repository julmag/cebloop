# @title Supplements
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.animation import FuncAnimation
import scipy
from sklearn.model_selection import train_test_split
import pandas as pd


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






def pearsons (targets, output):
    # Pearson correlation coefficient
    x_r, x_p = scipy.stats.pearsonr(targets[:,0], output[:,0])
    y_r, y_p = scipy.stats.pearsonr(targets[:,1], output[:,1])

    return np.array((x_r, y_r))





# @title Data Generation



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




