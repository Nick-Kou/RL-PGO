from __future__ import print_function

from minisam import *
from minisam.sophus import *
import numpy as np
import math
import sys

import matplotlib
import matplotlib.pyplot as plt


###########################################################################################################################################################
#TODO USER ENTER THIS ONLY
path_to_use = 'runs/Toy_Pose_sac_v2_lstm_Training322'
number_of_evals = 10
#This option allows GN100 plot to also be stacked but if poor estimate will be too zoomed out and wont be able to see the other estimates.
#Therefore use the separate plotter script for this one and set to false for now
plot_GN100 = False
#either one or the other has to be true not both. If both false will just plot GT vs Noisy Estimate
RL_best = False
RL_GN10= False

#TODO GT path for output of Grid_GCN Training.py
GT_path = path_to_use + '/GT_graphs/GT_graph0.g2o'
Simulated_path = path_to_use + '/Simulated_graphs/Simulated_graph0.g2o'

if RL_best:
    #TODO GT path CHANGE TRAINING NUMBER AND LAST NUMBER OF EVAL for output of Grid_GCN Training.py
    Simulated_path_RL_best = path_to_use +'/Simulated_graphs/Simulated_graph_best_from'+str(number_of_evals)+'.g2o'
else:
    Simulated_path_RL_GN10 = path_to_use + '/Simulated_graphs/RL_GN10.g2o'

#TODO G2o estimate path
GN100_estimate_path = path_to_use +"/Simulated_graphs/GN100.g2o"
#######################################################################################################################################################
# plot SE2 with covariance
def plotSE2WithCov(pose, cov, vehicle_size=0.5, line_color='k', vehicle_color='r'):
    # plot vehicle
    p1 = pose.translation() + pose.so2() * np.array([1, 0]) * vehicle_size
    p2 = pose.translation() + pose.so2() * np.array([-0.5, -0.5]) * vehicle_size
    p3 = pose.translation() + pose.so2() * np.array([-0.5, 0.5]) * vehicle_size
    line = plt.Polygon([p1, p2, p3], closed=True, fill=True, edgecolor=line_color, facecolor=vehicle_color)
    plt.gca().add_line(line)
    # plot cov
    ps = []
    circle_count = 50
    for i in range(circle_count):
        t = float(i) / float(circle_count) * math.pi * 2.0
        cp = pose.translation() + np.matmul(cov[0:2, 0:2], np.array([math.cos(t), math.sin(t)]))
        ps.append(cp)
    line = plt.Polygon(ps, closed=True, fill=False, edgecolor=line_color)
    plt.gca().add_line(line)

def plotSE2_matplot(pose, vehicle_size=0.5, line_color='k', vehicle_color='r'):
    # plot vehicle
    p1 = pose.translation() + pose.so2() * np.array([1, 0]) * vehicle_size
    p2 = pose.translation() + pose.so2() * np.array([-0.5, -0.5]) * vehicle_size
    p3 = pose.translation() + pose.so2() * np.array([-0.5, 0.5]) * vehicle_size
    line = plt.Polygon([p1, p2, p3], closed=True, fill=True, edgecolor=line_color, facecolor=vehicle_color)
    plt.gca().add_line(line)


def plot2DPoseGraphResult(ax, graph, variables, color, linewidth=1):
    lines = []
    for factor in graph:
        # only plot between factor
        if factor.__class__.__name__ == BetweenFactor_SE2_.__name__:
            keys = factor.keys()
            p1 = variables.at_SE2_(keys[0]).translation()
            p2 = variables.at_SE2_(keys[1]).translation()
            lines.append([p1, p2])
    lc = matplotlib.collections.LineCollection(lines, colors=color, linewidths=linewidth)
    ax.add_collection(lc)
    plt.axis('equal')
if __name__ == '__main__':

    GT_graph = FactorGraph()
    GT_initials = Variables()

    Simulated_graph = FactorGraph()
    Simulated_initials = Variables()
    if RL_best:
        Simulated_graph_RL_best = FactorGraph()
        Simulated_initials_RL_best = Variables()
    else:
        RL_GN10_graph = FactorGraph()
        RL_GN10_initials = Variables()

    GN100_graph = FactorGraph()
    GN100_initials = Variables()

    # load .g2o file
    file3d = loadG2O(GT_path, GT_graph, GT_initials);
    Standardize_Pose_Graph(GT_graph, GT_initials)
    file3d = loadG2O(Simulated_path, Simulated_graph, Simulated_initials);
    Standardize_Pose_Graph(Simulated_graph, Simulated_initials)
    if RL_best:
        file3d = loadG2O(Simulated_path_RL_best, Simulated_graph_RL_best, Simulated_initials_RL_best);
        Standardize_Pose_Graph(Simulated_graph_RL_best, Simulated_initials_RL_best)
    if RL_GN10:
        file3d = loadG2O(Simulated_path_RL_GN10, RL_GN10_graph, RL_GN10_initials);
        Standardize_Pose_Graph(RL_GN10_graph, RL_GN10_initials)
    if plot_GN100:
        file3d = loadG2O(GN100_estimate_path, GN100_graph, GN100_initials);
        Standardize_Pose_Graph(GN100_graph, GN100_initials)
    lossprior = ScaleLoss.Scale(1.0)
    GT_Prior = GT_initials.at(key('x', 0))
    GT_graph.add(PriorFactor(key('x', 0), GT_Prior, lossprior));

    if plot_GN100:
        GN100_graph.add(PriorFactor(key('x', 0), GT_Prior, lossprior));

    fig, ax = plt.subplots()
    for i in range(GT_initials.size()):
        plotSE2_matplot(GT_initials.at(key('x', i)), vehicle_color='g')
    for i in range(Simulated_initials.size()):
        plotSE2_matplot(Simulated_initials.at(key('x', i)), vehicle_color='r')
    if RL_best:
        for i in range(Simulated_initials_RL_best.size()):
            plotSE2_matplot(Simulated_initials_RL_best.at(key('x', i)), vehicle_color='k')
    if RL_GN10:
        for i in range(RL_GN10_initials.size()):
            plotSE2_matplot(RL_GN10_initials.at(key('x', i)), vehicle_color='m')

    if plot_GN100:
        for i in range(GN100_initials.size()):
            plotSE2_matplot(GN100_initials.at(key('x', i)), vehicle_color='b')


    plot2DPoseGraphResult(ax, GT_graph, GT_initials, 'g', linewidth=1)
    plot2DPoseGraphResult(ax, Simulated_graph, Simulated_initials, 'r', linewidth=1)
    if RL_best:
        plot2DPoseGraphResult(ax, Simulated_graph_RL_best, Simulated_initials_RL_best, 'k', linewidth=1)
    if RL_GN10:
        plot2DPoseGraphResult(ax, RL_GN10_graph, RL_GN10_initials, 'm', linewidth=1)

    if plot_GN100:
        plot2DPoseGraphResult(ax, GN100_graph, GN100_initials, 'b', linewidth=1)

    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')

    if not RL_best and not RL_GN10 and not plot_GN100:
        ax.set_title('Ground Truth (Green) vs Initial Noisy Estimate (Red)')
    if RL_best and plot_GN100:
        ax.set_title('Ground Truth (Green) vs Initial (Red) vs RL Estimate (Black) vs GN100 (Blue)')
    if RL_GN10 and RL_best:
        ax.set_title('Ground Truth (Green) vs Initial (Red) RL+GN10 (Magenta)')
    if RL_best and not plot_GN100:
        ax.set_title('Ground Truth (Green) vs Initial (Red) vs RL Estimate (Black)')
    if RL_GN10 and not plot_GN100:
        ax.set_title('Ground Truth (Green) vs Initial (Red) RL+GN10 (Magenta)')
    #ax.set_title('Ground Truth (Green) vs Initial Noisy Estimate (Red)')
    #ax.set_title('Grid World Environment')
    plt.show()
