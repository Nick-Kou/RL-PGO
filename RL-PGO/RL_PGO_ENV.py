import numpy as np
import random
import os
from gym.spaces.box import Box
from minisam import *
from minisam.sophus import *
import dgl
import torch as th



# GPU = False
# device_idx = 0
# if GPU:
#     device = th.device("cuda:" + str(device_idx) if th.cuda.is_available() else "cpu")
# else:
#     device = th.device("cpu")
#
device_idx = 0
device = th.device("cuda:" + str(device_idx) if th.cuda.is_available() else "cpu")
class PoseGraph:
    def __init__(self,boundx,boundy,num_of_nodes,path,trans_noise_,rot_noise_,inter_dist,sqrt_loss,change_noise,loop_close,prob_of_loop,state_dim,frob,reward_numerator_scale,rot_action_range):
        # Global variables
        bound_area = [boundx, boundy]
        self.rot_action_range = rot_action_range
        self.reward_numerator_scale = reward_numerator_scale
        self.frob = frob
        self.state_dim = state_dim
        self.path = path

        self.num_of_nodes= num_of_nodes
        self.times_changed_env = 0

        self.trans_noise_ = trans_noise_
        self.rot_noise_ = rot_noise_
        self.inter_dist = inter_dist
        self.sqrt_loss = sqrt_loss
        self.change_noise = change_noise
        self.loop_close = loop_close
        self.prob_of_loop = prob_of_loop
        self.stepLen= self.inter_dist * 1.0

        self.boundx=boundx
        self.boundy=boundy
        self.env_current_step=0
        path = self.path+"/Simulated_graphs"
        try:
            os.makedirs(path)
        except:
            pass
        path = self.path + "/GT_graphs"
        try:
            os.makedirs(path)
        except:
            pass
        GT_graph = FactorGraph()
        Simulated_graph = FactorGraph()
        if self.change_noise:

            trans_noise = round(np.random.uniform(low=0.01, high=10.0),3)
            #original trans noise
            #rot_noise = round(np.random.uniform(low=0.12, high=0.43),6)
            rot_noise = round(np.random.uniform(low=0.08, high=0.5), 3)
        else:
            trans_noise = trans_noise_
            rot_noise = rot_noise_



        odomLoss = DiagonalLoss.Sigmas(np.array([trans_noise, trans_noise, rot_noise]))
        loopLoss = DiagonalLoss.Sigmas(np.array([trans_noise, trans_noise, rot_noise]))
        loop_list = []


        Simulated_initials = Variables()
        GT_initials = Variables()
        GT_initials.add(key('x', 1), SE2(SO2(0), np.array([0, 0])))
        Simulated_initials.add(key('x', 1), SE2(SO2(0), np.array([0, 0])))
        if (self.num_of_nodes % 10 != 0) or (self.num_of_nodes <= 0):
            raise ValueError("Number of poses must be a multiple of 10 and greater than 0")

        Current_True_motion = SE2(SO2(0), np.array([self.stepLen, 0]))
        for i in range(1, 9):
            current_num_nodes = Simulated_initials.__len__()

            True_motion = Current_True_motion
            New_True_Pose = GT_initials.at(key('x', current_num_nodes)) * True_motion
            Noisy_motion = True_motion * (SE2.exp(np.array([np.random.normal(loc=0.0, scale=trans_noise), np.random.normal(loc=0.0, scale=trans_noise),np.random.normal(loc=0.0, scale=rot_noise)])))
            New_Simulated_Pose = Simulated_initials.at(key('x', current_num_nodes)) * Noisy_motion
            GT_initials.add(key('x', current_num_nodes + 1), New_True_Pose)
            Simulated_initials.add(key('x', current_num_nodes + 1), New_Simulated_Pose)

            True_Edge = (GT_initials.at(key('x', current_num_nodes))).inverse() * GT_initials.at(
                key('x', current_num_nodes + 1))

            Simulated_Edge = (Simulated_initials.at(key('x', current_num_nodes))).inverse() * Simulated_initials.at(
                key('x', current_num_nodes + 1))

            GT_graph.add(
                BetweenFactor(key('x', current_num_nodes), key('x', current_num_nodes + 1), True_Edge, odomLoss))
            Simulated_graph.add(
                BetweenFactor(key('x', current_num_nodes), key('x', current_num_nodes + 1), Simulated_Edge, odomLoss))
        turn_left_flag = True

        while ((Simulated_initials.__len__()) < self.num_of_nodes):
            current_num_nodes = Simulated_initials.__len__()

            if turn_left_flag == True:
                New_True_Motion = SE2(SO2(1.5708), np.array([self.stepLen, 0]))
                turn_left_flag = False
            else:
                New_True_Motion = SE2(SO2(-1.5708), np.array([self.stepLen, 0]))
                turn_left_flag = True
            for j in range(2):
                current_num_nodes = Simulated_initials.__len__()
                if current_num_nodes >= self.num_of_nodes:
                    break
                New_True_Pose = GT_initials.at(key('x', current_num_nodes)) * New_True_Motion
                Noisy_motion = New_True_Motion * (SE2.exp(np.array([np.random.normal(loc=0.0, scale=trans_noise), np.random.normal(loc=0.0, scale=trans_noise),np.random.normal(loc=0.0, scale=rot_noise)])))
                New_Simulated_Pose = Simulated_initials.at(key('x', current_num_nodes)) * Noisy_motion
                GT_initials.add(key('x', current_num_nodes + 1), New_True_Pose)
                Simulated_initials.add(key('x', current_num_nodes + 1), New_Simulated_Pose)

                True_Edge = (GT_initials.at(key('x', current_num_nodes))).inverse() * GT_initials.at(
                    key('x', current_num_nodes + 1))

                Simulated_Edge = (Simulated_initials.at(key('x', current_num_nodes))).inverse() * Simulated_initials.at(
                    key('x', current_num_nodes + 1))

                GT_graph.add(
                    BetweenFactor(key('x', current_num_nodes), key('x', current_num_nodes + 1), True_Edge, odomLoss))
                Simulated_graph.add(
                    BetweenFactor(key('x', current_num_nodes), key('x', current_num_nodes + 1), Simulated_Edge, odomLoss))
            current_num_nodes = Simulated_initials.__len__()
            loop_list.append([current_num_nodes, (current_num_nodes - 1)])
            seq = 3
            for i in range(1, 9):
                current_num_nodes = Simulated_initials.__len__()
                if current_num_nodes >= self.num_of_nodes:
                    break
                True_motion = Current_True_motion
                New_True_Pose = GT_initials.at(key('x', current_num_nodes)) * True_motion
                Noisy_motion = True_motion * (SE2.exp(np.array([np.random.normal(loc=0.0, scale=trans_noise), np.random.normal(loc=0.0, scale=trans_noise),np.random.normal(loc=0.0, scale=rot_noise)])))
                New_Simulated_Pose = Simulated_initials.at(key('x', current_num_nodes)) * Noisy_motion
                loop_list.append([current_num_nodes + 1, (current_num_nodes + 1) - seq])
                seq = seq + 2

                GT_initials.add(key('x', current_num_nodes + 1), New_True_Pose)
                Simulated_initials.add(key('x', current_num_nodes + 1), New_Simulated_Pose)

                True_Edge = (GT_initials.at(key('x', current_num_nodes))).inverse() * GT_initials.at(
                    key('x', current_num_nodes + 1))

                Simulated_Edge = (Simulated_initials.at(key('x', current_num_nodes))).inverse() * Simulated_initials.at(
                    key('x', current_num_nodes + 1))

                GT_graph.add(
                    BetweenFactor(key('x', current_num_nodes), key('x', current_num_nodes + 1), True_Edge, odomLoss))
                Simulated_graph.add(
                    BetweenFactor(key('x', current_num_nodes), key('x', current_num_nodes + 1), Simulated_Edge, odomLoss))


        if (self.loop_close == True) and (self.num_of_nodes > 10):
            list1 = random.sample(loop_list, (int(self.prob_of_loop * len(loop_list))))
            for i in list1:
                True_Edge = (GT_initials.at(key('x', i[0]))).inverse() * GT_initials.at(
                    key('x', i[1]))

                GT_graph.add(
                    BetweenFactor(key('x', i[0]), key('x', i[1]), True_Edge, loopLoss))
                Simulated_graph.add(
                    BetweenFactor(key('x', i[0]), key('x', i[1]), True_Edge, loopLoss))
        Standardize_Pose_Graph(GT_graph,GT_initials)
        Standardize_Pose_Graph(Simulated_graph, Simulated_initials)
        lossprior1 = ScaleLoss.Scale(1.0)
        self.GT_Prior = GT_initials.at(key('x', 0))
        GT_graph.add(PriorFactor(key('x', 0), self.GT_Prior, lossprior1));
        Simulated_graph.add(PriorFactor(key('x', 0), self.GT_Prior, lossprior1));
        self.var_ord = Simulated_initials.defaultVariableOrdering()


        #print("printing GT initials before loading")

        writeG2O(self.path + "/GT_graphs/GT_graph" + str(self.times_changed_env) + ".g2o", GT_graph, GT_initials)
        #Simulated_initials = addPoseNoiseSE2_except_anchor(Simulated_initials, trans_noise, rot_noise)

        writeG2O(self.path + "/Simulated_graphs/Simulated_graph" + str(self.times_changed_env) + ".g2o",Simulated_graph, Simulated_initials)
        print("printing initial simulated weighted squared norm error before loading")
        #print(Simulated_graph.errorSquaredNorm(Simulated_initials) / 2)
        print(Simulated_graph.errorSquaredNorm(Simulated_initials))
        print("printing initial GT weighted squared norm error before loading")
        #print(GT_graph.errorSquaredNorm(GT_initials) / 2)
        print(GT_graph.errorSquaredNorm(GT_initials))

        initial_meas = [[1, 0], [0, 1]]
        rot_meas = get_so2_meas(Simulated_graph)
        rot_meas.append(initial_meas)
        self.lidx = []
        self.ridx = []
        self.left_var_ord_for_stepping = []
        self.right_var_ord_for_stepping= []
        info = []
        var_ord_list = list(self.var_ord)
        for iter_1, factor in enumerate(Simulated_graph):
            if factor.__class__.__name__ == BetweenFactor_SE2_.__name__:
                keys = factor.keys()
                self.lidx.append(keyIndex(keys[0]))
                self.ridx.append(keyIndex(keys[1]))
                if keys[0] in var_ord_list:
                    self.left_var_ord_for_stepping.append(var_ord_list.index(keys[0]) * 3)

                if keys[1] in var_ord_list:
                    self.right_var_ord_for_stepping.append(var_ord_list.index(keys[1]) * 3)

                loss_function = factor.lossFunction().getter()
                if loss_function.size == 3:
                    # 1/sigma for rotation
                    info.append(loss_function[2])

                else:
                    info.append(loss_function[2][2])
        self.lidx.append(0)
        self.ridx.append(0)
        info.append(1.0)
        self.left_var_ord_for_stepping.append(0)
        self.right_var_ord_for_stepping.append(0)

        searchval = 0
        self.lidx_for_step = np.array(self.lidx)
        self.ridx_for_step = np.array(self.ridx)
        zero_in_left = np.where(self.lidx_for_step == searchval)[0]
        zero_in_right = np.where(self.ridx_for_step == searchval)[0]
        zero_in_left = np.append(zero_in_left, zero_in_right)
        zero_in_left = list(set(zero_in_left.tolist()))
        zero_in_left.sort(reverse=True)
        self.factor_graph_list_index = [i for i in range(len(self.lidx_for_step))]
        self.lidx_for_step = self.lidx_for_step.tolist()
        self.ridx_for_step = self.ridx_for_step.tolist()
        for value in (zero_in_left):
            del self.lidx_for_step[value]
            del self.ridx_for_step[value]
            del self.factor_graph_list_index[value]
            del self.left_var_ord_for_stepping[value]
            del self.right_var_ord_for_stepping[value]


        self.h_graph = dgl.graph((th.tensor(self.lidx).to(device), th.tensor(self.ridx).to(device)))
        self.h_graph.ndata['cost'] = th.zeros(self.h_graph.num_nodes(), 1).to(device)

        self.h_graph.ndata['rot_v'] = th.tensor(np.array(get_rotation_features(Simulated_initials))).to(device)


        self.h_graph.edata['rot_meas'] = th.tensor(np.array(rot_meas)).to(device)

        self.h_graph.edata['sq_info'] = th.tensor(info).to(device)

        self.h_graph.edata['lidx'] = th.tensor(self.lidx).to(device)
        self.h_graph.edata['ridx'] = th.tensor(self.ridx).to(device)
        # minus the prior and the first between factor 0-1 (which connects the anchor)
        self.num_btw_fct = len(self.factor_graph_list_index)

        self.trans_noise= trans_noise
        self.rot_noise = rot_noise

        self.global_GT_graph = GT_graph
        self.global_Simulated_graph = Simulated_graph
        self.global_GT_initials = GT_initials
        self.global_Simulated_initials = Simulated_initials

        self.GT_graph = self.global_GT_graph
        self.Simulated_graph = self.global_Simulated_graph
        self.GT_initials = self.global_GT_initials
        self.Simulated_initials = self.global_Simulated_initials

        
        self.initial_cost = self.Simulated_graph.get_all_errors(self.Simulated_initials)

        if self.sqrt_loss:
            if self.frob:
                self.initial_cost_rot = np.sqrt(self.Simulated_graph.frob_rot_err_sq_sum)
            else:
                self.initial_cost_rot = np.sqrt(self.Simulated_graph.non_weighted_rot_sq_sum)

        else:
            if self.frob:
                self.initial_cost_rot = self.Simulated_graph.frob_rot_err_sq_sum
            else:
                self.initial_cost_rot = self.Simulated_graph.non_weighted_rot_sq_sum

        self.best_cost = self.initial_cost
        self.best_cost_rot = self.initial_cost_rot
        self.best_initials = self.Simulated_initials
        self.current_cost = None
        self.current_cost_rot = None
        self.num_actions = 6
        self.num_actions_rot = 2



        self.num_observations = (3*(self.GT_graph.size()))+ 2
        
        self.rot_num_observations = (self.state_dim + 1)
        #######################################################################################
        self.action_space = Box(-1, 1, [self.num_actions])
        self.rot_action_space = Box(-1, 1, [self.num_actions_rot])
        self.observation_space = Box(-1000, 1000, [self.num_observations])
        self.rot_observation_space = Box(-1000, 1000, [self.rot_num_observations])




    def load_static_graph(self):
        filename_GT = self.path + "/GT_graph" + str(self.times_changed_env) + ".g2o"
        filename_Simulated = self.path + "/Simulated_graph" + str(self.times_changed_env) + ".g2o"
        loaded_graph_GT = FactorGraph()
        loaded_initials_GT = Variables()
        loaded_graph_Simulated = FactorGraph()
        loaded_initials_Simulated = Variables()
        file2dgt = loadG2O(filename_GT, loaded_graph_GT, loaded_initials_GT);
        file2dsi = loadG2O(filename_Simulated, loaded_graph_Simulated, loaded_initials_Simulated);

        # add a prior factor to first pose to fix the whole system
        lossprior = ScaleLoss.Scale(1.0)
        loaded_graph_GT.add(PriorFactor(key('x', 1), loaded_initials_GT.at(key('x', 1)), lossprior));
        loaded_graph_Simulated.add(PriorFactor(key('x', 1), loaded_initials_Simulated.at(key('x', 1)), lossprior));
        self.global_GT_graph = loaded_graph_GT
        self.global_Simulated_graph = loaded_graph_Simulated
        self.global_GT_initials = loaded_initials_GT
        self.global_Simulated_initials = loaded_initials_Simulated

        self.GT_graph = self.global_GT_graph
        self.Simulated_graph = self.global_Simulated_graph
        self.GT_initials = self.global_GT_initials
        self.Simulated_initials = self.global_Simulated_initials

    def every_third_Element(self,a):
        return np.array(a[2::3])


    def apply_action_range(self, action):
        list = []
        for i in range(len(action)):
            j = i + 1
            if j % 3 == 0:
                list.append(np.array([5, 5, 1]))
        array = np.array(list)
        array = array.reshape(len(action), )
        final_action = array * action
        return final_action

    def get_state(self):

        row=self.btw_fact_matrix[(self.env_current_step)%(self.num_btw_fct)]

      

        return np.append((self.Simulated_graph.get_residual_state(self.Simulated_initials)),row[0:2])

    def get_state_rot(self):

        
        rot_residuals = self.every_third_Element(self.Simulated_graph.get_residual_state(self.Simulated_initials))

        


        return np.append(rot_residuals,self.Simulated_graph[(self.env_current_step) % (self.num_btw_fct)].error(self.Simulated_initials)[2])


    def step(self, action, sparse_reward):
        # Get events and check if the user has closed the window
        final_action = np.zeros(self.num_of_nodes*3)
        row = self.btw_fact_matrix[(self.env_current_step-1) % (self.num_btw_fct)]
        final_action[int(row[2]):int(row[2])+3] = action[0:3]
        final_action[int(row[3]):int(row[3]) + 3] = action[3:6]


        self.Simulated_initials = self.Simulated_initials.retract(final_action, self.var_ord)
        #self.current_cost = self.Simulated_graph.errorSquaredNorm(self.Simulated_initials) / 2
        self.current_cost = self.Simulated_graph.errorSquaredNorm(self.Simulated_initials)
        if self.current_cost < self.best_cost:
            self.best_cost = self.current_cost
            self.best_initials = self.Simulated_initials
        distance2goal = self.current_cost

        if sparse_reward:
            if distance2goal < 0.01:
                reward = 20
            else:
                reward = -1
        else:
            # verison 1: inverse
            reward_numerator = 100.0
            reward = (reward_numerator / (distance2goal + 1))
            if distance2goal< 0.0001:
                reward = reward +25
                if distance2goal<0.00001:
                    reward = reward+25
                    if distance2goal <0.000001:
                        reward = reward+25
                        if distance2goal <0.0000001:
                            reward = reward+25
                            if distance2goal < 0.00000001:
                                reward = reward + 25
                                if distance2goal < 0.000000001:
                                    reward = reward + 25
            # version 2: negative
            # reward = -distance2goal
        #### Removed done statement so that tensor issues don't occur (if eposide finishes have to fill the remaining steps by 0 instead of being empty)
        # if self.current_cost<=0.01:
        #     done = 1
        # else:
        #     done = 0
        done = 0
        state_after_pertubation = self.get_state()

        return state_after_pertubation, reward, done

    def rot_step(self, action, sparse_reward):
        # Get events and check if the user has closed the window
        final_action = np.zeros(self.num_of_nodes*3)
        final_action[int(self.left_var_ord_for_stepping[(self.env_current_step-1) % (self.num_btw_fct)])+2] = action[0] * self.rot_action_range
        final_action[int(self.right_var_ord_for_stepping[(self.env_current_step - 1) % (self.num_btw_fct)])+2] = action[1] *self.rot_action_range


        self.Simulated_initials = self.Simulated_initials.retract(final_action, self.var_ord)
        

        self.Simulated_graph.get_all_errors(self.Simulated_initials)


        if self.sqrt_loss:
            if self.frob:
                self.current_cost_rot = np.sqrt(self.Simulated_graph.frob_rot_err_sq_sum)
            else:
                self.current_cost_rot = np.sqrt(self.Simulated_graph.non_weighted_rot_sq_sum)

        else:
            if self.frob:
                self.current_cost_rot = self.Simulated_graph.frob_rot_err_sq_sum
            else:
                self.current_cost_rot = self.Simulated_graph.non_weighted_rot_sq_sum

        if self.current_cost_rot < self.best_cost_rot:
            self.best_cost_rot = self.current_cost_rot
            self.best_initials = self.Simulated_initials
        distance2goal = self.current_cost_rot

        if sparse_reward:
            if distance2goal < 0.01:
                reward = 20
            else:
                reward = -1
        else:
            # verison 1: inverse
            reward_numerator = 100.0 *self.reward_numerator_scale
            reward = (reward_numerator / (distance2goal + 1))
            if distance2goal< 0.0001:
                reward = reward +25
                if distance2goal<0.00001:
                    reward = reward+25
                    if distance2goal <0.000001:
                        reward = reward+25
                        if distance2goal <0.0000001:
                            reward = reward+25
                            if distance2goal < 0.00000001:
                                reward = reward + 25
                                if distance2goal < 0.000000001:
                                    reward = reward + 25
            # version 2: negative
            # reward = -distance2goal
        #### Removed done statement so that tensor issues don't occur (if eposide finishes have to fill the remaining steps by 0 instead of being empty)
        # if self.current_cost<=0.01:
        #     done = 1
        # else:
        #     done = 0
        done = 0
        self.h_graph.ndata['rot_v'] = th.tensor(np.array(get_rotation_features(self.Simulated_initials))).to(device)
        #state_after_pertubation = self.get_state_rot()

        return self.h_graph, reward, done



    def reset(self):
        self.GT_graph = self.global_GT_graph
        self.Simulated_graph = self.global_Simulated_graph
        self.GT_initials = self.global_GT_initials
        self.Simulated_initials = self.global_Simulated_initials


        self.var_ord = self.Simulated_initials.defaultVariableOrdering()
        
        self.initial_cost = self.Simulated_graph.errorSquaredNorm(self.Simulated_initials)
        self.best_cost = self.initial_cost
        self.best_initials = self.Simulated_initials
        

        state_list = self.get_state()
        return state_list

    def reset_rot(self):
        self.GT_graph = self.global_GT_graph
        self.Simulated_graph = self.global_Simulated_graph
        self.GT_initials = self.global_GT_initials
        self.Simulated_initials = self.global_Simulated_initials


        self.var_ord = self.Simulated_initials.defaultVariableOrdering()
        #self.Simulated_graph.get_all_errors(self.Simulated_initials) / 2
        self.Simulated_graph.get_all_errors(self.Simulated_initials)
        if self.sqrt_loss:
            if self.frob:
                self.initial_cost_rot = np.sqrt(self.Simulated_graph.frob_rot_err_sq_sum)
            else:
                self.initial_cost_rot = np.sqrt(self.Simulated_graph.non_weighted_rot_sq_sum)

        else:
            if self.frob:
                self.initial_cost_rot = self.Simulated_graph.frob_rot_err_sq_sum
            else:
                self.initial_cost_rot = self.Simulated_graph.non_weighted_rot_sq_sum


        self.best_cost_rot = self.initial_cost_rot
        self.best_initials = self.Simulated_initials
        
        self.h_graph.ndata['rot_v'] = th.tensor(np.array(get_rotation_features(self.Simulated_initials))).to(device)

        return self.h_graph



    # TODO instead of using changedGT graph and initials just use global instead to avoid expensive copies and memory
    def change_target(self,robust,save):
        self.times_changed_env= self.times_changed_env+1
        changed_GT_graph = FactorGraph()
        changed_Simulated_graph = FactorGraph()
        if self.change_noise:

            trans_noise = round(np.random.uniform(low=0.01, high=10.0),3)
            #original trans noise
            #rot_noise = round(np.random.uniform(low=0.12, high=0.43),6)
            rot_noise = round(np.random.uniform(low=0.08, high=0.5), 3)
        else:
            trans_noise = self.trans_noise_
            rot_noise = self.rot_noise_
        

        odomLoss = DiagonalLoss.Sigmas(np.array([trans_noise, trans_noise, rot_noise]))
        loopLoss = DiagonalLoss.Sigmas(np.array([trans_noise, trans_noise, rot_noise]))
        loop_list = []


        changed_Simulated_initials = Variables()
        changed_GT_initials = Variables()
        changed_GT_initials.add(key('x', 1), SE2(SO2(0), np.array([0, 0])))
        changed_Simulated_initials.add(key('x', 1), SE2(SO2(0), np.array([0, 0])))
        if (self.num_of_nodes % 10 != 0) or (self.num_of_nodes <= 0):
            raise ValueError("Number of poses must be a multiple of 10 and greater than 0")

        Current_True_motion = SE2(SO2(0), np.array([self.stepLen, 0]))
        for i in range(1, 9):
            current_num_nodes = changed_Simulated_initials.__len__()

            True_motion = Current_True_motion
            New_True_Pose = changed_GT_initials.at(key('x', current_num_nodes)) * True_motion
            Noisy_motion = True_motion * (SE2.exp(np.array(
                [np.random.normal(loc=0.0, scale=trans_noise), np.random.normal(loc=0.0, scale=trans_noise),
                 np.random.normal(loc=0.0, scale=rot_noise)])))
            New_Simulated_Pose = changed_Simulated_initials.at(key('x', current_num_nodes)) * Noisy_motion
            changed_GT_initials.add(key('x', current_num_nodes + 1), New_True_Pose)
            changed_Simulated_initials.add(key('x', current_num_nodes + 1), New_Simulated_Pose)

            True_Edge = (changed_GT_initials.at(key('x', current_num_nodes))).inverse() * changed_GT_initials.at(
                key('x', current_num_nodes + 1))

            Simulated_Edge = (changed_Simulated_initials.at(
                key('x', current_num_nodes))).inverse() * changed_Simulated_initials.at(
                key('x', current_num_nodes + 1))

            changed_GT_graph.add(
                BetweenFactor(key('x', current_num_nodes), key('x', current_num_nodes + 1), True_Edge, odomLoss))
            changed_Simulated_graph.add(
                BetweenFactor(key('x', current_num_nodes), key('x', current_num_nodes + 1), Simulated_Edge, odomLoss))
        turn_left_flag = True

        while ((changed_Simulated_initials.__len__()) < self.num_of_nodes):
            current_num_nodes = changed_Simulated_initials.__len__()

            if turn_left_flag == True:
                New_True_Motion = SE2(SO2(1.5708), np.array([self.stepLen, 0]))
                turn_left_flag = False
            else:
                New_True_Motion = SE2(SO2(-1.5708), np.array([self.stepLen, 0]))
                turn_left_flag = True
            for j in range(2):
                current_num_nodes = changed_Simulated_initials.__len__()
                if current_num_nodes >= self.num_of_nodes:
                    break
                New_True_Pose = changed_GT_initials.at(key('x', current_num_nodes)) * New_True_Motion
                Noisy_motion = New_True_Motion * (SE2.exp(np.array(
                    [np.random.normal(loc=0.0, scale=trans_noise), np.random.normal(loc=0.0, scale=trans_noise),
                     np.random.normal(loc=0.0, scale=rot_noise)])))
                New_Simulated_Pose = changed_Simulated_initials.at(key('x', current_num_nodes)) * Noisy_motion
                changed_GT_initials.add(key('x', current_num_nodes + 1), New_True_Pose)
                changed_Simulated_initials.add(key('x', current_num_nodes + 1), New_Simulated_Pose)

                True_Edge = (changed_GT_initials.at(key('x', current_num_nodes))).inverse() * changed_GT_initials.at(
                    key('x', current_num_nodes + 1))

                Simulated_Edge = (changed_Simulated_initials.at(
                    key('x', current_num_nodes))).inverse() * changed_Simulated_initials.at(
                    key('x', current_num_nodes + 1))

                changed_GT_graph.add(
                    BetweenFactor(key('x', current_num_nodes), key('x', current_num_nodes + 1), True_Edge, odomLoss))
                changed_Simulated_graph.add(
                    BetweenFactor(key('x', current_num_nodes), key('x', current_num_nodes + 1), Simulated_Edge, odomLoss))
            current_num_nodes = changed_Simulated_initials.__len__()
            loop_list.append([current_num_nodes, (current_num_nodes - 1)])
            seq = 3
            for i in range(1, 9):
                current_num_nodes = changed_Simulated_initials.__len__()
                if current_num_nodes >= self.num_of_nodes:
                    break
                True_motion = Current_True_motion
                New_True_Pose = changed_GT_initials.at(key('x', current_num_nodes)) * True_motion
                Noisy_motion = True_motion * (SE2.exp(np.array(
                    [np.random.normal(loc=0.0, scale=trans_noise), np.random.normal(loc=0.0, scale=trans_noise),
                     np.random.normal(loc=0.0, scale=rot_noise)])))
                New_Simulated_Pose = changed_Simulated_initials.at(key('x', current_num_nodes)) * Noisy_motion
                loop_list.append([current_num_nodes + 1, (current_num_nodes + 1) - seq])
                seq = seq + 2

                changed_GT_initials.add(key('x', current_num_nodes + 1), New_True_Pose)
                changed_Simulated_initials.add(key('x', current_num_nodes + 1), New_Simulated_Pose)

                True_Edge = (changed_GT_initials.at(key('x', current_num_nodes))).inverse() * changed_GT_initials.at(
                    key('x', current_num_nodes + 1))

                Simulated_Edge = (changed_Simulated_initials.at(
                    key('x', current_num_nodes))).inverse() * changed_Simulated_initials.at(
                    key('x', current_num_nodes + 1))

                changed_GT_graph.add(
                    BetweenFactor(key('x', current_num_nodes), key('x', current_num_nodes + 1), True_Edge, odomLoss))
                changed_Simulated_graph.add(
                    BetweenFactor(key('x', current_num_nodes), key('x', current_num_nodes + 1), Simulated_Edge, odomLoss))


        if (self.loop_close == True) and (self.num_of_nodes > 10):
            list1 = random.sample(loop_list, (int(self.prob_of_loop * len(loop_list))))
            for i in list1:
                True_Edge = (changed_GT_initials.at(key('x', i[0]))).inverse() * changed_GT_initials.at(
                    key('x', i[1]))

                changed_GT_graph.add(
                    BetweenFactor(key('x', i[0]), key('x', i[1]), True_Edge, loopLoss))
                changed_Simulated_graph.add(
                    BetweenFactor(key('x', i[0]), key('x', i[1]), True_Edge, loopLoss))
        Standardize_Pose_Graph(changed_GT_graph, changed_GT_initials)
        Standardize_Pose_Graph(changed_Simulated_graph, changed_Simulated_initials)
        lossprior1 = ScaleLoss.Scale(1.0)
        self.GT_Prior = changed_GT_initials.at(key('x', 0))
        changed_GT_graph.add(PriorFactor(key('x', 0), self.GT_Prior, lossprior1));
        changed_Simulated_graph.add(PriorFactor(key('x', 0), self.GT_Prior, lossprior1));
        self.var_ord = changed_Simulated_initials.defaultVariableOrdering()

        
        if save:
            writeG2O(self.path + "/GT_graphs/GT_graph" + str(self.times_changed_env) + ".g2o", changed_GT_graph, changed_GT_initials)
            writeG2O(self.path + "/Simulated_graphs/Simulated_graph" + str(self.times_changed_env) + ".g2o",changed_Simulated_graph, changed_Simulated_initials)

        initial_meas = [[1, 0], [0, 1]]
        rot_meas = get_so2_meas(changed_Simulated_graph)
        rot_meas.append(initial_meas)
        self.lidx = []
        self.ridx = []
        self.left_var_ord_for_stepping = []
        self.right_var_ord_for_stepping = []
        info = []
        var_ord_list = list(self.var_ord)
        for iter_1, factor in enumerate(changed_Simulated_graph):
            if factor.__class__.__name__ == BetweenFactor_SE2_.__name__:
                keys = factor.keys()
                self.lidx.append(keyIndex(keys[0]))
                self.ridx.append(keyIndex(keys[1]))
                if keys[0] in var_ord_list:
                    self.left_var_ord_for_stepping.append(var_ord_list.index(keys[0]) * 3)

                if keys[1] in var_ord_list:
                    self.right_var_ord_for_stepping.append(var_ord_list.index(keys[1]) * 3)

                loss_function = factor.lossFunction().getter()
                if loss_function.size == 3:
                    # 1/sigma for rotation
                    info.append(loss_function[2])

                else:
                    info.append(loss_function[2][2])
        self.lidx.append(0)
        self.ridx.append(0)
        info.append(1.0)
        self.left_var_ord_for_stepping.append(0)
        self.right_var_ord_for_stepping.append(0)

        searchval = 0
        self.lidx_for_step = np.array(self.lidx)
        self.ridx_for_step = np.array(self.ridx)
        zero_in_left = np.where(self.lidx_for_step == searchval)[0]
        zero_in_right = np.where(self.ridx_for_step == searchval)[0]
        zero_in_left = np.append(zero_in_left, zero_in_right)
        zero_in_left = list(set(zero_in_left.tolist()))
        zero_in_left.sort(reverse=True)
        self.factor_graph_list_index = [i for i in range(len(self.lidx_for_step))]
        self.lidx_for_step = self.lidx_for_step.tolist()
        self.ridx_for_step = self.ridx_for_step.tolist()
        for value in (zero_in_left):
            del self.lidx_for_step[value]
            del self.ridx_for_step[value]
            del self.factor_graph_list_index[value]
            del self.left_var_ord_for_stepping[value]
            del self.right_var_ord_for_stepping[value]

        self.h_graph = dgl.graph((th.tensor(self.lidx).to(device), th.tensor(self.ridx).to(device)))
        self.h_graph.ndata['cost'] = th.zeros(self.h_graph.num_nodes(), 1).to(device)
        self.h_graph.ndata['rot_v'] = th.tensor(get_rotation_features(changed_Simulated_initials)).to(device)

        self.h_graph.edata['rot_meas'] = th.tensor(rot_meas).to(device)
        self.h_graph.edata['sq_info'] = th.tensor(info).to(device)
        self.h_graph.edata['lidx'] = th.tensor(self.lidx).to(device)
        self.h_graph.edata['ridx'] = th.tensor(self.ridx).to(device)
        # minus the prior and the first between factor 0-1 (which connects the anchor)
        self.num_btw_fct = len(self.factor_graph_list_index)

        self.trans_noise = trans_noise
        self.rot_noise = rot_noise

        self.global_GT_graph = changed_GT_graph
        self.global_Simulated_graph = changed_Simulated_graph
        self.global_GT_initials = changed_GT_initials
        self.global_Simulated_initials = changed_Simulated_initials

        ##### Dont need below because reset should be called each time after env.change_target

        # self.GT_graph = self.global_GT_graph
        # self.Simulated_graph = self.global_Simulated_graph
        # self.GT_initials = self.global_GT_initials
        # self.Simulated_initials = self.global_Simulated_initials

class Load_Pose_Graph(PoseGraph):
    def __init__(self,boundx,boundy,path,sqrt_loss,state_dim,frob,reward_numerator_scale,rot_action_range):
        # Global variables
        bound_area = [boundx, boundy]
        self.rot_action_range = rot_action_range
        self.reward_numerator_scale = reward_numerator_scale
        self.frob = frob
        self.state_dim = state_dim
        self.path = path

        self.sqrt_loss = sqrt_loss

        self.times_changed_env = 0
        self.boundx=boundx
        self.boundy=boundy
        self.env_current_step = 0


        filename_GT = self.path + "/GT_graphs/GT_graph" + str(self.times_changed_env) + ".g2o"
        filename_Simulated = self.path + "/Simulated_graphs/Simulated_graph" + str(self.times_changed_env) + ".g2o"
        loaded_graph_GT = FactorGraph()
        loaded_initials_GT = Variables()
        loaded_graph_Simulated = FactorGraph()
        loaded_initials_Simulated = Variables()
        file2dgt = loadG2O(filename_GT, loaded_graph_GT, loaded_initials_GT);
        file2dsi = loadG2O(filename_Simulated, loaded_graph_Simulated, loaded_initials_Simulated);
        Standardize_Pose_Graph(loaded_graph_GT, loaded_initials_GT)
        Standardize_Pose_Graph(loaded_graph_Simulated, loaded_initials_Simulated)
        # add a prior factor to first pose to fix the whole system
        lossprior = ScaleLoss.Scale(1.0)
        self.GT_Prior1 = loaded_initials_GT.at(key('x', 0))
        loaded_graph_GT.add(PriorFactor(key('x', 0), self.GT_Prior1, lossprior));
        loaded_graph_Simulated.add(PriorFactor(key('x', 0), self.GT_Prior1, lossprior));

        self.var_ord = loaded_initials_Simulated.defaultVariableOrdering()
        self.num_of_nodes = loaded_initials_GT.size()

        # Only need to use 1 filename and 1 observation as all observations and GT and simulated have the same rot and trans noise, not the case for real world datasets (don't have GT)

        initial_meas = [[1, 0], [0, 1]]
        rot_meas = get_so2_meas(loaded_graph_Simulated)
        rot_meas.append(initial_meas)
        self.lidx = []
        self.ridx = []
        self.left_var_ord_for_stepping = []
        self.right_var_ord_for_stepping = []
        info = []
        var_ord_list = list(self.var_ord)
        for iter_1, factor in enumerate(loaded_graph_Simulated):
            if factor.__class__.__name__ == BetweenFactor_SE2_.__name__:
                keys = factor.keys()
                self.lidx.append(keyIndex(keys[0]))
                self.ridx.append(keyIndex(keys[1]))
                if keys[0] in var_ord_list:
                    self.left_var_ord_for_stepping.append(var_ord_list.index(keys[0]) * 3)

                if keys[1] in var_ord_list:
                    self.right_var_ord_for_stepping.append(var_ord_list.index(keys[1]) * 3)

                loss_function = factor.lossFunction().getter()
                if loss_function.size == 3:
                    # 1/sigma for rotation
                    info.append(loss_function[2])

                else:
                    info.append(loss_function[2][2])
        self.lidx.append(0)
        self.ridx.append(0)
        info.append(1.0)
        self.left_var_ord_for_stepping.append(0)
        self.right_var_ord_for_stepping.append(0)

        searchval = 0
        self.lidx_for_step = np.array(self.lidx)
        self.ridx_for_step = np.array(self.ridx)
        zero_in_left = np.where(self.lidx_for_step == searchval)[0]
        zero_in_right = np.where(self.ridx_for_step == searchval)[0]
        zero_in_left = np.append(zero_in_left, zero_in_right)
        zero_in_left = list(set(zero_in_left.tolist()))
        zero_in_left.sort(reverse=True)
        self.factor_graph_list_index = [i for i in range(len(self.lidx_for_step))]
        self.lidx_for_step = self.lidx_for_step.tolist()
        self.ridx_for_step = self.ridx_for_step.tolist()
        for value in (zero_in_left):
            del self.lidx_for_step[value]
            del self.ridx_for_step[value]
            del self.factor_graph_list_index[value]
            del self.left_var_ord_for_stepping[value]
            del self.right_var_ord_for_stepping[value]

        self.h_graph = dgl.graph((th.tensor(self.lidx).to(device), th.tensor(self.ridx).to(device)))
        self.h_graph.ndata['cost'] = th.zeros(self.h_graph.num_nodes(), 1).to(device)
        self.h_graph.ndata['rot_v'] = th.tensor(np.array(get_rotation_features(loaded_initials_Simulated))).to(device)

        self.h_graph.edata['rot_meas'] = th.tensor(np.array(rot_meas)).to(device)
        self.h_graph.edata['sq_info'] = th.tensor(info).to(device)
        self.h_graph.edata['lidx'] = th.tensor(self.lidx).to(device)
        self.h_graph.edata['ridx'] = th.tensor(self.ridx).to(device)
        # minus the prior and the first between factor 0-1 (which connects the anchor)
        self.num_btw_fct = len(self.factor_graph_list_index)

        print("Printing loaded simulated graph weighted error square norm")
        #print(loaded_graph_Simulated.errorSquaredNorm(loaded_initials_Simulated) / 2)
        print(loaded_graph_Simulated.errorSquaredNorm(loaded_initials_Simulated))
        print("Printing loaded GT graph weighted error square norm")
        #print(loaded_graph_GT.errorSquaredNorm(loaded_initials_GT) / 2)
        print(loaded_graph_GT.errorSquaredNorm(loaded_initials_GT))

        self.global_GT_graph = loaded_graph_GT
        self.global_Simulated_graph = loaded_graph_Simulated
        self.global_GT_initials = loaded_initials_GT
        self.global_Simulated_initials = loaded_initials_Simulated

        self.GT_graph = self.global_GT_graph
        self.Simulated_graph = self.global_Simulated_graph
        self.GT_initials = self.global_GT_initials
        self.Simulated_initials = self.global_Simulated_initials

        # Doesn't matter what we set it to because once we call reset then this value will change accordingly to if the agent is rotation or translation or regular
        #self.initial_cost = self.Simulated_graph.get_all_errors(self.Simulated_initials) / 2
        self.initial_cost = self.Simulated_graph.get_all_errors(self.Simulated_initials)
        if self.sqrt_loss:
            if self.frob:
                self.initial_cost_rot = np.sqrt(self.Simulated_graph.frob_rot_err_sq_sum)
            else:
                self.initial_cost_rot = np.sqrt(self.Simulated_graph.non_weighted_rot_sq_sum)

        else:
            if self.frob:
                self.initial_cost_rot = self.Simulated_graph.frob_rot_err_sq_sum
            else:
                self.initial_cost_rot = self.Simulated_graph.non_weighted_rot_sq_sum



        self.best_cost = self.initial_cost
        self.best_cost_rot = self.initial_cost_rot
        self.best_initials = self.Simulated_initials
        self.current_cost = None
        self.current_cost_rot = None


        self.num_actions = 6
        self.num_actions_rot = 2
        
        self.num_observations = (3 * (self.GT_graph.size())) + 2
        
        self.rot_num_observations = (self.state_dim + 1)


        #######################################################################################
        self.action_space = Box(-1, 1, [self.num_actions])
        self.rot_action_space = Box(-1, 1, [self.num_actions_rot])
        self.observation_space = Box(-1000, 1000, [self.num_observations])
        self.rot_observation_space = Box(-1000, 1000, [self.rot_num_observations])

