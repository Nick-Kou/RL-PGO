import numpy as np
import math
from minisam import *
from minisam.sophus import *





class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean([x], axis=0)
        batch_var = np.var([x], axis=0)
        batch_count = 1
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

def Convert(string):
    li = list(string.split(" "))
    return li

def plotSE2(pose, vehicle_size=0.5, ):
    # plot vehicle
    p1 = pose.translation() + pose.so2() * np.array([1, 0]) * vehicle_size
    p2 = pose.translation() + pose.so2() * np.array([-0.5, -0.5]) * vehicle_size
    p3 = pose.translation() + pose.so2() * np.array([-0.5, 0.5]) * vehicle_size
    midpoint = (p2 + p3) / 2
    p4 = (midpoint + p1) / 2
    # p4 = p1-midpoint
    # temp = p4*2/3
    # p4= p4-temp

    p1 = np.append(p1, 0)
    p2 = np.append(p2, 0)
    p3 = np.append(p3, 0)
    p4 = np.append(p4, 0)
    p1 = np.reshape(p1, (-1, 3))
    p2 = np.reshape(p2, (-1, 3))
    p3 = np.reshape(p3, (-1, 3))
    p4 = np.reshape(p4, (-1, 3))
    return p1, p2, p3, p4


def updated_trajectory_plotter(global_simulated_initials):
    for i in range(global_simulated_initials.__len__()):

        p5, p6, p7, p8 = plotSE2(global_simulated_initials.at(key('x', i + 1)))
        if i == 0:

            Simulated_trajectory = p8

            Simulated_trajectory = np.append(Simulated_trajectory, p5, axis=0)
            Simulated_trajectory = np.append(Simulated_trajectory, p6, axis=0)
            Simulated_trajectory = np.append(Simulated_trajectory, p7, axis=0)
            Simulated_trajectory = np.append(Simulated_trajectory, p5, axis=0)

        else:

            Simulated_trajectory = np.append(Simulated_trajectory, p8, axis=0)
            Simulated_trajectory = np.append(Simulated_trajectory, p5, axis=0)
            Simulated_trajectory = np.append(Simulated_trajectory, p6, axis=0)
            Simulated_trajectory = np.append(Simulated_trajectory, p7, axis=0)
            Simulated_trajectory = np.append(Simulated_trajectory, p5, axis=0)
    return Simulated_trajectory

def plotSE2_new(pose,vehicle_size=0.5 ):
    # plot vehicle
    p1 = pose.translation() + pose.so2() * np.array([1, 0]) * vehicle_size
    p2 = pose.translation() + pose.so2() * np.array([-0.5, -0.5]) * vehicle_size
    p3 = pose.translation() + pose.so2() * np.array([-0.5, 0.5]) * vehicle_size

    # p4 = p1-midpoint
    # temp = p4*2/3
    # p4= p4-temp


    p1 = np.append(p1,0)
    p2 = np.append(p2, 0)
    p3 = np.append(p3, 0)

    p1 = np.reshape(p1, (-1, 3))
    p2 = np.reshape(p2, (-1, 3))
    p3 = np.reshape(p3, (-1, 3))

    return p1,p2,p3


def updated_trajectory_plotter_for_anchor_zero_idx(initials,graph):
    list_of_poses = []
    list_of_btw_factors = []
    for i in range(initials.__len__()):

        p5, p6, p7 = plotSE2_new(initials.at(key('x', i )))
        plot1 = p5
        plot1 = np.append(plot1, p6, axis=0)
        plot1 = np.append(plot1, p7, axis=0)
        plot1 = np.append(plot1, p5, axis=0)
        list_of_poses.append(plot1)

    for factor in graph:
        # only plot between factor
        if factor.__class__.__name__ == BetweenFactor_SE2_.__name__:
            keys = factor.keys()
            p1 = initials.at_SE2_(keys[0]).translation()
            p2 = initials.at_SE2_(keys[1]).translation()
            p1 = np.append(p1, 0)
            p2 = np.append(p2, 0)
            p1 = np.reshape(p1, (-1, 3))
            p2 = np.reshape(p2, (-1, 3))
            plot2 = p1
            plot2 = np.append(plot2, p2, axis=0)
            list_of_btw_factors.append(plot2)

    return list_of_poses, list_of_btw_factors

def addPoseNoiseSE2_Rounded(initials, trans_noise, rot_noise):
    list = []

    for variable_num in range(initials.__len__()):
        list.append(np.array(
            [round(np.random.normal(loc=0.0, scale=trans_noise),3), round(np.random.normal(loc=0.0, scale=trans_noise),3),
             round(np.random.normal(loc=0.0, scale=rot_noise),3)]))
    noise_array = np.array(list)
    noise_array = noise_array.reshape(initials.__len__() * 3, )

    var_ord = initials.defaultVariableOrdering()
    initials = initials.retract(noise_array, var_ord)
    for variable_num in range(initials.__len__()):
        trans_true = initials.__getitem__(key('x', variable_num + 1)).translation()
        rot_true = initials.__getitem__(key('x', variable_num + 1)).log()
        rounded_x = round(trans_true[0],6)
        rounded_y = round(trans_true[1], 6)
        rounded_theta = round(rot_true[2], 6)
        Rounded_Var = SE2(SO2(rounded_theta), np.array([rounded_x, rounded_y]))
        initials.update(key('x', variable_num+1), Rounded_Var)
    return initials

def addPoseNoiseSE2(initials, trans_noise, rot_noise):
    list = []
    for variable_num in range(initials.__len__()):
        list.append(np.array(
            [round(np.random.normal(loc=0.0, scale=trans_noise),3),
             round(np.random.normal(loc=0.0, scale=trans_noise),3),
             round(np.random.normal(loc=0.0, scale=rot_noise),3)]))
    noise_array = np.array(list)
    noise_array = noise_array.reshape(initials.__len__() * 3, )

    var_ord = initials.defaultVariableOrdering()
    initials = initials.retract(noise_array, var_ord)
    return initials

def addPoseNoiseSE2_except_anchor(initials, trans_noise, rot_noise):
    list1 = []
    for variable_num in range(initials.__len__()):
        list1.append(np.array(
            [round(np.random.normal(loc=0.0, scale=trans_noise),3),
             round(np.random.normal(loc=0.0, scale=trans_noise),3),
             round(np.random.normal(loc=0.0, scale=rot_noise),3)]))
    noise_array = np.array(list1)
    noise_array = noise_array.reshape(initials.__len__() * 3, )

    var_ord = initials.defaultVariableOrdering()
    var_ord_list = list(var_ord)
    idx = var_ord_list.index(key('x',0)) *3
    noise_array[idx] = 0.0
    noise_array[idx+1] = 0.0
    noise_array[idx+2] = 0.0
    initials = initials.retract(noise_array, var_ord)
    return initials

def get_retraction_values(initials, trans_noise, rot_noise):
    list = []
    for variable_num in range(initials.__len__()):
        list.append(np.array(
            [round(np.random.normal(loc=0.0, scale=trans_noise), 3),
             round(np.random.normal(loc=0.0, scale=trans_noise), 3),
             round(np.random.normal(loc=0.0, scale=rot_noise), 3)]))
    noise_array = np.array(list)
    noise_array = noise_array.reshape(initials.__len__() * 3, )

    return noise_array


def apply_action_range(action):
    list = []
    for i in range(len(action)):
        j = i + 1
        if j % 3 == 0:
            list.append(np.array([5, 5, 1]))
    array = np.array(list)
    array = array.reshape(len(action), )
    final_action = array * action
    return final_action


def return_action_range(num_of_actions, trans_range, rot_range):
    list = []
    for i in range(num_of_actions):
        j = i + 1
        if j % 3 == 0:
            list.append(np.array([trans_range, trans_range, rot_range]))
    array = np.array(list)
    array = array.reshape(num_of_actions, )

    return array

