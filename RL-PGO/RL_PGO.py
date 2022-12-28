'''
Soft Actor-Critic version 2
using target Q instead of V net: 2 Q net, 2 target Q net, 1 policy net
add alpha loss compared with version 1
paper: https://arxiv.org/pdf/1812.05905.pdf
'''

import OpenGL.GL as gl
#import pangolin
#import wandb
from minisam import *
from minisam.sophus import *


import os
import csv
from torch.utils import tensorboard
from common.buffers import *
from common.policy_networks import *
from Sac_trainer_agents import SAC_Trainer_rotation_w_grad_option_GCN
from Matplotlib_Pose_Plotter import plotSE2_matplot,plot2DPoseGraphResult
import matplotlib.pyplot as plt
import dgl



plt.style.use('ggplot')

from RL_PGO_ENV import PoseGraph, Load_Pose_Graph
from Pose_graph_env_utils import updated_trajectory_plotter_for_anchor_zero_idx

import argparse
import time
# GPU = False
# #GPU = True
# device_idx = 0
# if GPU:
#     device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
# else:
#     device = torch.device("cpu")
device_idx = 0
device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
print(device)

parser = argparse.ArgumentParser(description='Train or test on pose graph env.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)
parser.add_argument('--load', dest='load', action='store_true', default=False)
parser.add_argument('--no-load', dest='load', action='store_false', default=False)
parser.add_argument('--frob', dest='frob', action='store_true', default=False)
parser.add_argument('--no-frob', dest='frob', action='store_false', default=False)
parser.add_argument('--grad', dest='grad', action='store_true', default=False)
parser.add_argument('--no-grad', dest='grad', action='store_false', default=False)
parser.add_argument('--change', dest='change', action='store_true', default=False)
parser.add_argument('--no-change', dest='change', action='store_false', default=False)
parser.add_argument('--gui', dest='gui', action='store_true', default=False)
parser.add_argument('--no-gui', dest='gui', action='store_false', default=False)
parser.add_argument('--sqrt_loss', dest='sqrt_loss', action='store_true', default=False)
parser.add_argument('--no_sqrt_loss', dest='sqrt_loss', action='store_false', default=False)
parser.add_argument('--lr', type=float,default=None, required=True)
parser.add_argument('--gamma', type=float,default=None, required=True)
parser.add_argument('--grad_value', type=float,default=None, required=True)
parser.add_argument('--change_noise', dest='change_noise', action='store_true', default=False)
parser.add_argument('--no_change_noise', dest='change_noise', action='store_false', default=False)
parser.add_argument('--trans_noise', type=float,default=None, required=True)
parser.add_argument('--rot_noise', type=float,default=None, required=True)
parser.add_argument('--state_dim', type=int,default=None, required=True)
parser.add_argument('--reward_numerator_scale', type=float,default=None, required=True)
parser.add_argument('--reward_update_scale', type=float,default=None, required=True)
parser.add_argument('--loop_close', dest='loop_close', action='store_true', default=False)
parser.add_argument('--no_loop_close', dest='loop_close', action='store_false', default=False)
parser.add_argument('--inter_dist', type=int,default=None, required=True)

parser.add_argument('--prob_of_loop', type=float,default=None, required=True)
parser.add_argument('--freq_of_change', type=int,default=None, required=True)
parser.add_argument('--hidden_dim', type=int,default=None, required=True)
parser.add_argument('--training_name', type=str,default=None, required=True)
parser.add_argument('--simulated_style', type=str,default=None, required=False)
parser.add_argument('--num_nodes', type=int,default=None, required=True)
parser.add_argument('--rot_action_range', type=float,default=None, required=True)
parser.add_argument('--batch_size', type=int,default=None, required=True)
parser.add_argument('--update_itr', type=int,default=None, required=True)
parser.add_argument('--max_steps', type=int,default=None, required=True)
parser.add_argument('--max_episodes', type=int,default=None, required=True)
parser.add_argument('--max_evaluation_episodes', type=int,default=None, required=True)


args = parser.parse_args()

replay_buffer_size = 1e6
replay_buffer = ReplayBufferLSTM2_GCN(replay_buffer_size)



path = 'runs/'+args.training_name

try:
    os.makedirs(path)
except:
    pass
path2 = path + "/TensorboardLog_rot"
try:
    os.makedirs(path2)
except:
    pass

# choose env
#TODO these values set the box width and height for visualization only
global_boundx = int(0)
global_boundy = int(0)

# hyper-parameters for RL training
max_episodes = args.max_episodes
max_evaluation_episodes = args.max_evaluation_episodes
batch_size = args.batch_size
explore_steps = 0  # for action sampling in the beginning of training
update_itr = args.update_itr
AUTO_ENTROPY = True
DETERMINISTIC = False
hidden_dim = args.hidden_dim
rewards = []
number_of_nodes = args.num_nodes
trans_noise_ = args.trans_noise
rot_noise_ = args.rot_noise
state_dim = args.state_dim
reward_numerator_scale = args.reward_numerator_scale
reward_update_scale = args.reward_update_scale
inter_dist = args.inter_dist
sqrt_loss = args.sqrt_loss
change_noise = args.change_noise
loop_close = args.loop_close
prob_of_loop = args.prob_of_loop
frob = args.frob
gamma= args.gamma
lr = args.lr
grad = args.grad
grad_value = args.grad_value
r_act_range=args.rot_action_range
if __name__ == '__main__':
    global_Simulated_poses = []
    global_GT_poses = []
    global_Simulated_btw_factors = []
    global_GT_btw_factors = []
    global_Simulated_poses.append(np.zeros(shape=(1, 3)))
    global_GT_poses.append(np.zeros(shape=(1, 3)))
    global_Simulated_btw_factors.append(np.zeros(shape=(1, 3)))
    global_GT_btw_factors.append(np.zeros(shape=(1, 3)))

    if args.train:
        # wandb.init(project='', entity="")
        # wandb.run.name = args.training_name
        # wandb.run.save()
        if args.change:
            # TODO CHANGE max steps and number of poses and action range accordingly
            if args.load:
                env = Load_Pose_Graph(global_boundx, global_boundy, path,sqrt_loss,state_dim,frob,reward_numerator_scale,r_act_range)
                action_space = env.rot_action_space
                state_space = env.rot_observation_space
                action_dim = action_space.shape[0]
                rot_action_range = args.rot_action_range
                max_steps = args.max_steps * env.num_btw_fct
                frequency = args.freq_of_change
            else:
                env = PoseGraph(global_boundx, global_boundy, number_of_nodes, path,trans_noise_,rot_noise_,inter_dist,sqrt_loss,change_noise,loop_close,prob_of_loop,state_dim,frob,reward_numerator_scale,r_act_range)
                max_steps = args.max_steps *env.num_btw_fct
                #action_space = env.action_space
                action_space = env.rot_action_space
                state_space = env.rot_observation_space
                action_dim = action_space.shape[0]
                #action_range = return_action_range(env.num_actions, 5, 1)
                frequency = args.freq_of_change
                rot_action_range = args.rot_action_range

            sac_trainer_rotation = SAC_Trainer_rotation_w_grad_option_GCN(replay_buffer, state_space, action_space,lr,grad,grad_value, hidden_dim=hidden_dim,
                                      action_range=rot_action_range)
            # wandb.watch(models=sac_trainer_rotation.soft_q_net1)
            # wandb.watch(models=sac_trainer_rotation.soft_q_net2)
            # wandb.watch(models=sac_trainer_rotation.target_soft_q_net1)
            # wandb.watch(models=sac_trainer_rotation.target_soft_q_net2)
            # wandb.watch(models=sac_trainer_rotation.policy_net)


            # training loop
            best_episode_reward = None
            w = tensorboard.SummaryWriter(log_dir=path2)
            all_steps = 0
            robust = False

            for eps in range(max_episodes):
                if eps%frequency == 0:
                    env.change_target(False,False)
                env.env_current_step = 0
                observation=env.reset_rot()
                last_action = rot_action_range * (env.rot_action_space.sample())
                episode_observation = []
                episode_action = []
                episode_last_action = []
                episode_reward = []
                episode_next_observation = []
                episode_done = []
                episode_agent_residual= []
                episode_agent_next_residual=[]
                hidden_out = (torch.zeros([1, 1, hidden_dim], dtype=torch.float).to(device), torch.zeros([1, 1, hidden_dim],
                                                                                                     dtype=torch.float).to(device))  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)

                for step in range(max_steps):
                    residual_ = env.Simulated_graph[env.factor_graph_list_index[(env.env_current_step) % (env.num_btw_fct)]].error(env.Simulated_initials)[2]
                    state = np.append((sac_trainer_rotation.RotConv(observation).squeeze(1).cpu().detach().numpy()), residual_)
                    env.env_current_step = step+1
                    all_steps = all_steps + 1
                    hidden_in = hidden_out
                    action, hidden_out = sac_trainer_rotation.policy_net.get_action(state, last_action, hidden_in,
                                                                           deterministic=DETERMINISTIC)

                    next_observation, reward, done = env.rot_step(action, False)
                    next_residual_ = env.Simulated_graph[env.factor_graph_list_index[(env.env_current_step) % (env.num_btw_fct)]].error(env.Simulated_initials)[2]

                    if step == 0:
                        ini_hidden_in = hidden_in
                        ini_hidden_out = hidden_out
                    episode_observation.append(observation)
                    episode_action.append(action)
                    episode_last_action.append(last_action)
                    episode_reward.append(reward)
                    episode_next_observation.append(next_observation)
                    episode_done.append(done)
                    episode_agent_residual.append(residual_)
                    episode_agent_next_residual.append(next_residual_)

                    observation = next_observation
                    last_action = action

                    if len(replay_buffer) > batch_size:
                        for i in range(update_itr):
                            loss1, loss2, loss3, loss4 = sac_trainer_rotation.update(path,batch_size, reward_scale=reward_update_scale,
                                                                            auto_entropy=AUTO_ENTROPY,
                                                                            target_entropy=-1. * action_dim,gamma=gamma)
                            # w.add_scalar("average_q_loss_rot", loss1, global_step=all_steps)
                            # w.add_scalar("policy_loss_rot", loss2, global_step=all_steps)
                            # w.add_scalar("alpha_loss_rot", loss3, global_step=all_steps)
                            # w.add_scalar("total_loss_rot", loss4, global_step=all_steps)
                            # wandb.log({
                            #     "average_q_loss_rot": loss1,
                            #     "policy_loss_rot": loss2,
                            #     "alpha_loss_rot": loss3,
                            #     "total_loss_rot": loss4, })

                    if step % 20 == 0 and eps > 1:
                        sac_trainer_rotation.save_checkpoint(path, rewards)

                    if done:
                        break
                replay_buffer.push(ini_hidden_in, ini_hidden_out, dgl.batch(episode_observation), episode_action, episode_last_action,
                                   episode_reward, dgl.batch(episode_next_observation), episode_done,episode_agent_residual,episode_agent_next_residual)

                if len(rewards) == 0:
                    mean_100ep_reward = 0
                elif len(rewards) < 100:
                    mean_100ep_reward = np.mean(rewards)
                else:
                    mean_100ep_reward = np.mean(rewards[-100:])

                if len(rewards) > 100:
                    if best_episode_reward == None or mean_100ep_reward > best_episode_reward:
                        best_episode_reward = mean_100ep_reward
                        sac_trainer_rotation.save_best_model(path)

                print('Episode: ', eps, '| Episode Reward: ', np.sum(episode_reward))
                with open(path + "/Episode-Steps-Cost_Rotation.csv", 'a') as csvfile:
                    # creating a csv writer object
                    csvwriter = csv.writer(csvfile)

                    F_ = [len(rewards), step, env.initial_cost_rot, env.current_cost_rot, env.best_cost_rot]

                    csvwriter.writerow(F_)
                    csvfile.close()
                # w.add_scalar("reward/episode_reward_rot", np.sum(episode_reward), global_step=eps)
                # wandb_episode_reward = np.sum(episode_reward)
                # wandb.log({"Episode_Reward": wandb_episode_reward, 'episode': eps})

                rewards.append(np.sum(episode_reward))
            sac_trainer_rotation.save_checkpoint(path, rewards)
            w.close()
        ## for arg.test ensure that the number of nodes in line 377 is the same as the one in training, and also the max steps
        # can be changed to any multiple of 2 does not have to be the same as the one in training.
        # max_steps/2 is the number of adjustments each pose faces in an episode.
        else:
            # TODO CHANGE max steps and number of poses and action range accordingly
            if args.load:
                env = Load_Pose_Graph(global_boundx, global_boundy, path,sqrt_loss,state_dim,frob,reward_numerator_scale,r_act_range)
                action_space = env.rot_action_space
                state_space = env.rot_observation_space
                action_dim = action_space.shape[0]
                rot_action_range = args.rot_action_range
                max_steps = args.max_steps * env.num_btw_fct
                frequency = args.freq_of_change
            else:
                env = PoseGraph(global_boundx, global_boundy, number_of_nodes, path,trans_noise_,rot_noise_,inter_dist,sqrt_loss,change_noise,loop_close,prob_of_loop,state_dim,frob,reward_numerator_scale,r_act_range)
                max_steps = args.max_steps * env.num_btw_fct
                
                action_space = env.rot_action_space
                state_space = env.rot_observation_space
                action_dim = action_space.shape[0]
                
                frequency = args.freq_of_change
                rot_action_range = args.rot_action_range

            sac_trainer_rotation = SAC_Trainer_rotation_w_grad_option_GCN(replay_buffer, state_space, action_space, lr,
                                                                      grad, grad_value, hidden_dim=hidden_dim,
                                                                      action_range=rot_action_range)
            # wandb.watch(models=sac_trainer_rotation.soft_q_net1)
            # wandb.watch(models=sac_trainer_rotation.soft_q_net2)
            # wandb.watch(models=sac_trainer_rotation.target_soft_q_net1)
            # wandb.watch(models=sac_trainer_rotation.target_soft_q_net2)
            # wandb.watch(models=sac_trainer_rotation.policy_net)

            # training loop
            best_episode_reward = None
            w = tensorboard.SummaryWriter(log_dir=path2)
            all_steps = 0
            robust = False

            for eps in range(max_episodes):
                env.env_current_step = 0
                observation = env.reset_rot()
                last_action = rot_action_range * (env.rot_action_space.sample())
                episode_observation = []
                episode_action = []
                episode_last_action = []
                episode_reward = []
                episode_next_observation = []
                episode_done = []
                episode_agent_residual = []
                episode_agent_next_residual = []
                hidden_out = (torch.zeros([1, 1, hidden_dim], dtype=torch.float).to(device), torch.zeros([1, 1, hidden_dim],
                                                                                                     dtype=torch.float).to(device))  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)

                for step in range(max_steps):
                    residual_ = env.Simulated_graph[env.factor_graph_list_index[(env.env_current_step) % (env.num_btw_fct)]].error(env.Simulated_initials)[2]
                    state = np.append((sac_trainer_rotation.RotConv(observation).squeeze(1).cpu().detach().numpy()),residual_)
                    env.env_current_step = step + 1
                    all_steps = all_steps + 1
                    hidden_in = hidden_out
                    action, hidden_out = sac_trainer_rotation.policy_net.get_action(state, last_action, hidden_in,device,
                                                                                    deterministic=DETERMINISTIC)

                    next_observation, reward, done = env.rot_step(action, False)
                    next_residual_ = env.Simulated_graph[env.factor_graph_list_index[(env.env_current_step) % (env.num_btw_fct)]].error(env.Simulated_initials)[2]

                    if step == 0:
                        ini_hidden_in = hidden_in
                        ini_hidden_out = hidden_out
                    episode_observation.append(observation)
                    episode_action.append(action)
                    episode_last_action.append(last_action)
                    episode_reward.append(reward)
                    episode_next_observation.append(next_observation)
                    episode_done.append(done)
                    episode_agent_residual.append(residual_)
                    episode_agent_next_residual.append(next_residual_)

                    observation = next_observation
                    last_action = action

                    if len(replay_buffer) > batch_size:
                        for i in range(update_itr):
                            loss1, loss2, loss3, loss4 = sac_trainer_rotation.update(path, batch_size, reward_scale=reward_update_scale,
                                                                                     auto_entropy=AUTO_ENTROPY,
                                                                                     target_entropy=-1. * action_dim,gamma=gamma)
                            # w.add_scalar("average_q_loss_rot", loss1, global_step=all_steps)
                            # w.add_scalar("policy_loss_rot", loss2, global_step=all_steps)
                            # w.add_scalar("alpha_loss_rot", loss3, global_step=all_steps)
                            # w.add_scalar("total_loss_rot", loss4, global_step=all_steps)
                            # wandb.log({
                            #     "average_q_loss_rot": loss1,
                            #     "policy_loss_rot": loss2,
                            #     "alpha_loss_rot": loss3,
                            #     "total_loss_rot": loss4, })

                    if step % 20 == 0 and eps > 1:
                        sac_trainer_rotation.save_checkpoint(path, rewards)

                    if done:
                        break
                replay_buffer.push(ini_hidden_in, ini_hidden_out, dgl.batch(episode_observation), episode_action, episode_last_action,
                                   episode_reward, dgl.batch(episode_next_observation), episode_done,episode_agent_residual,episode_agent_next_residual)

                if len(rewards) == 0:
                    mean_100ep_reward = 0
                elif len(rewards) < 100:
                    mean_100ep_reward = np.mean(rewards)
                else:
                    mean_100ep_reward = np.mean(rewards[-100:])

                if len(rewards) > 100:
                    if best_episode_reward == None or mean_100ep_reward > best_episode_reward:
                        best_episode_reward = mean_100ep_reward
                        sac_trainer_rotation.save_best_model(path)

                print('Episode: ', eps, '| Episode Reward: ', np.sum(episode_reward))
                with open(path + "/Episode-Steps-Cost_Rotation.csv", 'a') as csvfile:
                    # creating a csv writer object
                    csvwriter = csv.writer(csvfile)

                    F_ = [len(rewards), step, env.initial_cost_rot, env.current_cost_rot, env.best_cost_rot]

                    csvwriter.writerow(F_)
                    csvfile.close()
                # w.add_scalar("reward/episode_reward_rot", np.sum(episode_reward), global_step=eps)
                # wandb_episode_reward = np.sum(episode_reward)
                # wandb.log({"Episode_Reward": wandb_episode_reward, 'episode': eps})

                rewards.append(np.sum(episode_reward))
            sac_trainer_rotation.save_checkpoint(path, rewards)
            w.close()
        ## for arg.test ensure that the number of nodes in line 377 is the same as the one in training, and also the max steps
        # can be changed to any multiple of 2 does not have to be the same as the one in training.
        # max_steps/2 is the number of adjustments each pose faces in an episode.

    if args.test:
        if args.gui:
            if args.change:
                if args.load:
                    env = Load_Pose_Graph(global_boundx, global_boundy, path,sqrt_loss,state_dim,frob,reward_numerator_scale,r_act_range)
                    action_space = env.rot_action_space
                    state_space = env.rot_observation_space
                    action_dim = action_space.shape[0]
                    action_range = args.rot_action_range
                    max_steps = args.max_steps * env.num_btw_fct
                    frequency = args.freq_of_change
                else:
                    env = PoseGraph(global_boundx, global_boundy, number_of_nodes, path,trans_noise_,rot_noise_,inter_dist,sqrt_loss,change_noise,loop_close,prob_of_loop,state_dim,frob,reward_numerator_scale,r_act_range)
                    max_steps = args.max_steps * env.num_btw_fct
                    # action_space = env.action_space
                    action_space = env.rot_action_space
                    state_space = env.rot_observation_space
                    action_dim = action_space.shape[0]
                    # action_range = return_action_range(env.num_actions, 5, 1)
                    frequency = args.freq_of_change
                    action_range = args.rot_action_range

                sac_trainer = SAC_Trainer_rotation_w_grad_option_GCN(replay_buffer, state_space, action_space, lr,
                                                                          grad, grad_value, hidden_dim=hidden_dim,
                                                                          action_range=action_range)


                pangolin.ParseVarsFile('app.cfg')

                #original
                #pangolin.CreateWindowAndBind('Main', 640, 480)

                pangolin.CreateWindowAndBind('Main', 640, 480)
                gl.glEnable(gl.GL_DEPTH_TEST)

                scam = pangolin.OpenGlRenderState(
                    #original
                    # pangolin.ProjectionMatrix(1040, 880, 420, 420, 320, 240, 0.2, 100),
                    # pangolin.ModelViewLookAt(0, 0, 75, 0, 0, 0, pangolin.AxisY))
                    # TODO adjust the 5040 and 7000 for zoom in and out
                    pangolin.ProjectionMatrix(5040, 7000, 420, 420, 320, 240, 0.2, 100),
                    pangolin.ModelViewLookAt(0, 0, 75, 0, 0, 0, pangolin.AxisY))

                handler3d = pangolin.Handler3D(scam)

                dcam = pangolin.CreateDisplay()
                #original
                #dcam.SetBounds(0.0, 1.0, 180 / 640., 1.0, -640.0 / 480.0)
                dcam.SetBounds(0.0, 1.0, 180 / 640., 1.0, -440.0 / 480.0)
                # dcam.SetBounds(pangolin.Attach(0.0),     pangolin.Attach(1.0),
                # pangolin.Attach.Pix(180), pangolin.Attach(1.0), -640.0/480.0)
                dcam.SetHandler(pangolin.Handler3D(scam))

                # panel = pangolin.CreatePanel('ui')
                # panel.SetBounds(0.0, 1.0, 0.0, 180 / 640.)
                panel = pangolin.CreatePanel('ui')
                panel.SetBounds(0.0, 1.0, 0.0, 180 / 640.)
                print("This is the Evaluation GUI, for automatic evaluation press New Episode for initial error print and inital graph visualization and then press Original Test button. For manual step by step use, press the New Episode Button and then the Step Testing button")
                Original_Test_button = pangolin.VarBool('ui.Original Test', value=False, toggle=False)
                Step_Testing_button = pangolin.VarBool('ui.Step Test', value=False, toggle=False)
                Matplotlib_button = pangolin.VarBool('ui.Matplotlib Save Best', value=False, toggle=False)
                New_Episode_button = pangolin.VarBool('ui.New Episode', value=False, toggle=False)
                Show_Best_button = pangolin.VarBool('ui.Show Best', value=False, toggle=False)
                Translational_Solver_button = pangolin.VarBool('ui.Translation Solver', value=False, toggle=False)
                global_step_counter = 0
                global_epsiode_counter = 0

                bound_box = [[-global_boundx, -global_boundy, 0], [-global_boundx, global_boundy, 0],
                             [global_boundx, global_boundy, 0], [global_boundx, -global_boundy, 0],
                             [-global_boundx, -global_boundy, 0]]
                sac_trainer.load_best_model(path,device)
                #sac_trainer.load_model(path)
                while not pangolin.ShouldQuit():
                    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
                    dcam.Activate(scam)
                    if pangolin.Pushed(Matplotlib_button):
                        fig, ax = plt.subplots()
                        for i in range(env.GT_initials.size()):
                            plotSE2_matplot(env.GT_initials.at(key('x', i )), vehicle_color='g')
                        for i in range(env.Simulated_initials.size()):
                            plotSE2_matplot(env.best_initials.at(key('x', i )), vehicle_color='r')

                        plot2DPoseGraphResult(ax, env.Simulated_graph, env.best_initials, 'r', linewidth=1)
                        plot2DPoseGraphResult(ax, env.GT_graph, env.GT_initials, 'g', linewidth=1)

                        ax.set_xlabel('X Position (m)')
                        ax.set_ylabel('Y Position (m)')
                        ax.set_title('Ground Truth (Green) vs RL Estimate (Red)')
                        plt.show()
                    if pangolin.Pushed(New_Episode_button):
                        if global_epsiode_counter == max_evaluation_episodes:
                            print("Max Evaluation Episodes Have Been Reached!")
                        else:

                            if global_epsiode_counter % frequency == 0:
                                env.change_target(False, False)

                            global_step_counter = 0
                            env.env_current_step = 0
                            observation = env.reset_rot()
                            #print("Initial graph error squared norm weighted cost is " + str(
                                #env.Simulated_graph.errorSquaredNorm(env.Simulated_initials) / 2))
                            print("Initial graph error squared norm weighted cost is " + str(
                                env.Simulated_graph.errorSquaredNorm(env.Simulated_initials)))
                            if sqrt_loss:
                                print("Initial non weighted Rotational Graph error sqrt  " + str(
                                    env.initial_cost_rot))
                            else:
                                print("Initial Non weighted Rotational Graph error squared sum  " + str(env.initial_cost_rot))

                            Simulated_poses,Simulated_btw_factors = updated_trajectory_plotter_for_anchor_zero_idx(env.Simulated_initials,env.Simulated_graph)
                            GT_poses,GT_btw_factors = updated_trajectory_plotter_for_anchor_zero_idx(env.GT_initials,env.GT_graph)
                            global_Simulated_poses = Simulated_poses
                            global_Simulated_btw_factors = Simulated_btw_factors
                            global_GT_poses = GT_poses
                            global_GT_btw_factors = GT_btw_factors

                            last_action = action_range * (env.rot_action_space.sample())
                            episode_reward = 0
                            hidden_out = (torch.zeros([1, 1, hidden_dim], dtype=torch.float).to(device),
                                          torch.zeros([1, 1, hidden_dim],
                                                      dtype=torch.float).to(device))  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
                            global_epsiode_counter = global_epsiode_counter + 1

                    if pangolin.Pushed(Step_Testing_button):
                        if global_step_counter == max_steps:
                            print("Max Steps Have Been Reached! Press New Episode To Restart")
                            episode_num = global_epsiode_counter - 1
                            if sqrt_loss:
                                print('Episode: ', episode_num, '| Episode Reward: ', episode_reward,
                                      '| Best Rotation Cost sqrt: ',
                                      env.best_cost_rot)
                            else:
                                print('Episode: ', episode_num, '| Episode Reward: ', episode_reward, '| Best Rotation Cost squared: ',
                                  env.best_cost_rot)
                        else:
                            residual_ = env.Simulated_graph[env.factor_graph_list_index[(env.env_current_step) % (env.num_btw_fct)]].error(env.Simulated_initials)[2]
                            state = np.append((sac_trainer.RotConv(observation).squeeze(1).cpu().detach().numpy()),residual_)
                            env.env_current_step = env.env_current_step + 1
                            hidden_in = hidden_out
                            action, hidden_out = sac_trainer.policy_net.get_action(state, last_action, hidden_in,device,
                                                                                   deterministic=True)
                            next_observation, reward, done = env.rot_step(action, False)
                            Simulated_poses, Simulated_btw_factors = updated_trajectory_plotter_for_anchor_zero_idx(env.Simulated_initials, env.Simulated_graph)
                            GT_poses, GT_btw_factors = updated_trajectory_plotter_for_anchor_zero_idx(env.GT_initials,
                                                                                                      env.GT_graph)
                            global_Simulated_poses = Simulated_poses
                            global_Simulated_btw_factors = Simulated_btw_factors
                            global_GT_poses = GT_poses
                            global_GT_btw_factors = GT_btw_factors

                            last_action = action
                            episode_reward += reward
                            observation = next_observation
                            global_step_counter = global_step_counter + 1
                            if sqrt_loss:
                                print('Step: ', global_step_counter, '| Non Weighted Rotation Current Cost sqrt: ',
                                      env.current_cost_rot, '| Reward: ', reward)
                            else:
                                print('Step: ', global_step_counter, '| Non Weighted Rotation Current Cost squared: ', env.current_cost_rot, '| Reward: ', reward)

                    if pangolin.Pushed(Show_Best_button):
                        Simulated_poses, Simulated_btw_factors = updated_trajectory_plotter_for_anchor_zero_idx(
                            env.best_initials, env.Simulated_graph)
                        GT_poses, GT_btw_factors = updated_trajectory_plotter_for_anchor_zero_idx(env.GT_initials,
                                                                                                  env.GT_graph)
                        global_Simulated_poses = Simulated_poses
                        global_Simulated_btw_factors = Simulated_btw_factors
                        global_GT_poses = GT_poses
                        global_GT_btw_factors = GT_btw_factors

                        if sqrt_loss:
                            print('Best SQRT Rotation Cost: ', env.best_cost_rot)
                        else:
                            print('Best Sum Squared Rotation Cost: ', env.best_cost_rot)
                        # print('Total weighted graph error square norm at the optimal rotation pose configuration: ',
                        #       (env.Simulated_graph.get_all_errors(env.best_initials)) / 2)
                        print('Total weighted graph error square norm at the optimal rotation pose configuration: ',
                              (env.Simulated_graph.get_all_errors(env.best_initials)) )

                    if pangolin.Pushed(Translational_Solver_button):
                        Trans_2D_Solver(env.Simulated_graph,env.best_initials)
                        Simulated_poses, Simulated_btw_factors = updated_trajectory_plotter_for_anchor_zero_idx(
                            env.best_initials, env.Simulated_graph)
                        GT_poses, GT_btw_factors = updated_trajectory_plotter_for_anchor_zero_idx(env.GT_initials,
                                                                                                  env.GT_graph)
                        global_Simulated_poses = Simulated_poses
                        global_Simulated_btw_factors = Simulated_btw_factors
                        global_GT_poses = GT_poses
                        global_GT_btw_factors = GT_btw_factors

                        # print('Total weighted graph error square norm: ',
                        #       (env.Simulated_graph.get_all_errors(env.best_initials)) / 2)
                        print('Total weighted graph error square norm: ',
                               (env.Simulated_graph.get_all_errors(env.best_initials)))
                    ############################################################################################################################
                    if pangolin.Pushed(Original_Test_button):

                        sac_trainer.load_best_model(path,device)
                        episode_reward_array = []
                        best_cost_rot_array = []
                        best_total_graph_cost_array = []
                        for eps in range(max_evaluation_episodes):
                            if pangolin.ShouldQuit():
                                break
                            if eps % frequency == 0:
                                env.change_target(False, False)
                            env.env_current_step = 0
                            observation = env.reset_rot()


                            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
                            dcam.Activate(scam)
                            Simulated_poses, Simulated_btw_factors = updated_trajectory_plotter_for_anchor_zero_idx(
                                env.Simulated_initials, env.Simulated_graph)
                            GT_poses, GT_btw_factors = updated_trajectory_plotter_for_anchor_zero_idx(env.GT_initials,
                                                                                                      env.GT_graph)
                            global_Simulated_poses = Simulated_poses
                            global_Simulated_btw_factors = Simulated_btw_factors
                            global_GT_poses = GT_poses
                            global_GT_btw_factors = GT_btw_factors
                            gl.glLineWidth(1.5)
                            gl.glColor3f(1.0, 1.0, 1.0)
                            pangolin.DrawLine(bound_box)
                            gl.glLineWidth(1)

                            gl.glColor3f(1, 0.0, 0.0)
                            for i in range(len(global_Simulated_poses)):
                                pangolin.DrawLine(global_Simulated_poses[i], point_size=0)
                            for i in range(len(global_Simulated_btw_factors)):
                                pangolin.DrawLine(global_Simulated_btw_factors[i], point_size=0)
                            gl.glColor3f(0.0, 1.0, 0.0)
                            for i in range(len(global_GT_poses)):
                                pangolin.DrawLine(global_GT_poses[i], point_size=0)
                            for i in range(len(global_GT_btw_factors)):
                                pangolin.DrawLine(global_GT_btw_factors[i], point_size=0)
                            pangolin.FinishFrame()

                            last_action = action_range * (env.rot_action_space.sample())
                            episode_reward = 0
                            hidden_out = (torch.zeros([1, 1, hidden_dim], dtype=torch.float).to(device),
                                          torch.zeros([1, 1, hidden_dim],
                                                      dtype=torch.float).to(device))  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)

                            for step in range(max_steps):
                                if pangolin.ShouldQuit():
                                    break
                                gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
                                dcam.Activate(scam)
                                residual_ = env.Simulated_graph[env.factor_graph_list_index[(env.env_current_step) % (env.num_btw_fct)]].error(env.Simulated_initials)[2]
                                state = np.append((sac_trainer.RotConv(observation).squeeze(1).cpu().detach().numpy()),residual_)
                                env.env_current_step = env.env_current_step + 1
                                hidden_in = hidden_out
                                action, hidden_out = sac_trainer.policy_net.get_action(state, last_action, hidden_in,device,
                                                                                       deterministic=True)
                                next_observation, reward, done = env.rot_step(action, False)
                                if step == max_steps-1:
                                    Trans_2D_Solver(env.Simulated_graph, env.best_initials)
                                    Simulated_poses, Simulated_btw_factors = updated_trajectory_plotter_for_anchor_zero_idx(
                                        env.best_initials, env.Simulated_graph)
                                    GT_poses, GT_btw_factors = updated_trajectory_plotter_for_anchor_zero_idx(
                                        env.GT_initials, env.GT_graph)
                                    global_Simulated_poses = Simulated_poses
                                    global_Simulated_btw_factors = Simulated_btw_factors
                                    global_GT_poses = GT_poses
                                    global_GT_btw_factors = GT_btw_factors

                                    gl.glLineWidth(1.5)
                                    gl.glColor3f(1.0, 1.0, 1.0)
                                    pangolin.DrawLine(bound_box)
                                    gl.glLineWidth(1)

                                    gl.glColor3f(1, 0.0, 0.0)
                                    for i in range(len(global_Simulated_poses)):
                                        pangolin.DrawLine(global_Simulated_poses[i], point_size=0)
                                    for i in range(len(global_Simulated_btw_factors)):
                                        pangolin.DrawLine(global_Simulated_btw_factors[i], point_size=0)
                                    gl.glColor3f(0.0, 1.0, 0.0)
                                    for i in range(len(global_GT_poses)):
                                        pangolin.DrawLine(global_GT_poses[i], point_size=0)
                                    for i in range(len(global_GT_btw_factors)):
                                        pangolin.DrawLine(global_GT_btw_factors[i], point_size=0)
                                    pangolin.FinishFrame()

                                else:
                                    Simulated_poses, Simulated_btw_factors = updated_trajectory_plotter_for_anchor_zero_idx(
                                        env.Simulated_initials, env.Simulated_graph)
                                    GT_poses, GT_btw_factors = updated_trajectory_plotter_for_anchor_zero_idx(
                                        env.GT_initials, env.GT_graph)
                                    global_Simulated_poses = Simulated_poses
                                    global_Simulated_btw_factors = Simulated_btw_factors
                                    global_GT_poses = GT_poses
                                    global_GT_btw_factors = GT_btw_factors

                                    gl.glLineWidth(1.5)
                                    gl.glColor3f(1.0, 1.0, 1.0)
                                    pangolin.DrawLine(bound_box)
                                    gl.glLineWidth(1)

                                    gl.glColor3f(1, 0.0, 0.0)
                                    for i in range(len(global_Simulated_poses)):
                                        pangolin.DrawLine(global_Simulated_poses[i], point_size=0)
                                    for i in range(len(global_Simulated_btw_factors)):
                                        pangolin.DrawLine(global_Simulated_btw_factors[i], point_size=0)
                                    gl.glColor3f(0.0, 1.0, 0.0)
                                    for i in range(len(global_GT_poses)):
                                        pangolin.DrawLine(global_GT_poses[i], point_size=0)
                                    for i in range(len(global_GT_btw_factors)):
                                        pangolin.DrawLine(global_GT_btw_factors[i], point_size=0)
                                    pangolin.FinishFrame()

                                last_action = action
                                episode_reward += reward
                                observation = next_observation

                            # graph_total_cost = ((env.Simulated_graph.get_all_errors(env.best_initials)) / 2)
                            graph_total_cost = (env.Simulated_graph.get_all_errors(env.best_initials))
                            episode_reward_array.append(episode_reward)
                            best_cost_rot_array.append(env.best_cost_rot)
                            best_total_graph_cost_array.append(graph_total_cost)
                            list1 = np.array(episode_reward_array)
                            list2 = np.array(best_cost_rot_array)
                            list3 = np.array(best_total_graph_cost_array)
                            if sqrt_loss:
                                print('Episode: ', eps, '| Episode Reward: ', episode_reward, '| Best Rotation Cost SQRT: ',
                                      env.best_cost_rot, '| Best Total Graph Cost(weighted error square norm): ',
                                      graph_total_cost)
                            else:
                                print('Episode: ', eps, '| Episode Reward: ', episode_reward,'| Best Rotation Cost Squared: ',env.best_cost_rot, '| Best Total Graph Cost(weighted error square norm): ', graph_total_cost)
                        print("Average Reward in "+str(max_evaluation_episodes)+" episodes", np.mean(list1))
                        if sqrt_loss:
                            print("Average Best Rotation SQRT loss(non weighted) in " + str(
                                max_evaluation_episodes) + " episodes", np.mean(list2))
                        else:
                            print("Average Best Rotation sum squared cost (non weighted) in " + str(
                                max_evaluation_episodes) + " episodes", np.mean(list2))


                        print("Average Best Graph error squared norm (weighted) in " + str(max_evaluation_episodes) + " episodes", np.mean(list3))

                    gl.glLineWidth(1.5)
                    gl.glColor3f(1.0, 1.0, 1.0)
                    pangolin.DrawLine(bound_box)
                    gl.glLineWidth(1)

                    gl.glColor3f(1, 0.0, 0.0)
                    for i in range(len(global_Simulated_poses)):
                        pangolin.DrawLine(global_Simulated_poses[i], point_size=0)
                    for i in range(len(global_Simulated_btw_factors)):
                        pangolin.DrawLine(global_Simulated_btw_factors[i], point_size=0)
                    gl.glColor3f(0.0, 1.0, 0.0)
                    for i in range(len(global_GT_poses)):
                        pangolin.DrawLine(global_GT_poses[i], point_size=0)
                    for i in range(len(global_GT_btw_factors)):
                        pangolin.DrawLine(global_GT_btw_factors[i], point_size=0)
                    pangolin.FinishFrame()
            else:
                if args.load:
                    env = Load_Pose_Graph(global_boundx, global_boundy, path,sqrt_loss,state_dim,frob,reward_numerator_scale,r_act_range)
                    action_space = env.rot_action_space
                    state_space = env.rot_observation_space
                    action_dim = action_space.shape[0]
                    action_range = args.rot_action_range
                    max_steps = args.max_steps * env.num_btw_fct
                    frequency = args.freq_of_change
                else:
                    env = PoseGraph(global_boundx, global_boundy, number_of_nodes, path,trans_noise_,rot_noise_,inter_dist,sqrt_loss,change_noise,loop_close,prob_of_loop,state_dim,frob,reward_numerator_scale,r_act_range)
                    max_steps = args.max_steps * env.num_btw_fct
                    # action_space = env.action_space
                    action_space = env.rot_action_space
                    state_space = env.rot_observation_space
                    action_dim = action_space.shape[0]
                    # action_range = return_action_range(env.num_actions, 5, 1)
                    frequency = args.freq_of_change
                    action_range = args.rot_action_range

                sac_trainer = SAC_Trainer_rotation_w_grad_option_GCN(replay_buffer, state_space, action_space, lr,
                                                                 grad, grad_value, hidden_dim=hidden_dim,
                                                                 action_range=action_range)

                pangolin.ParseVarsFile('app.cfg')

                # original
                # pangolin.CreateWindowAndBind('Main', 640, 480)

                pangolin.CreateWindowAndBind('Main', 640, 480)
                gl.glEnable(gl.GL_DEPTH_TEST)

                scam = pangolin.OpenGlRenderState(
                    # original
                    # pangolin.ProjectionMatrix(1040, 880, 420, 420, 320, 240, 0.2, 100),
                    # pangolin.ModelViewLookAt(0, 0, 75, 0, 0, 0, pangolin.AxisY))
                    #TODO adjust the 5040 and 7000 for zoom in and out
                    pangolin.ProjectionMatrix(5040, 7000, 420, 420, 320, 240, 0.2, 100),
                    pangolin.ModelViewLookAt(0, 0, 75, 0, 0, 0, pangolin.AxisY))

                handler3d = pangolin.Handler3D(scam)

                dcam = pangolin.CreateDisplay()
                # original
                # dcam.SetBounds(0.0, 1.0, 180 / 640., 1.0, -640.0 / 480.0)
                dcam.SetBounds(0.0, 1.0, 180 / 640., 1.0, -440.0 / 480.0)
                # dcam.SetBounds(pangolin.Attach(0.0),     pangolin.Attach(1.0),
                # pangolin.Attach.Pix(180), pangolin.Attach(1.0), -640.0/480.0)
                dcam.SetHandler(pangolin.Handler3D(scam))

                # panel = pangolin.CreatePanel('ui')
                # panel.SetBounds(0.0, 1.0, 0.0, 180 / 640.)
                panel = pangolin.CreatePanel('ui')
                panel.SetBounds(0.0, 1.0, 0.0, 180 / 640.)
                print(
                    "This is the Evaluation GUI, for automatic evaluation press New Episode for initial error print and inital graph visualization and then press Original Test button. For manual step by step use, press the New Episode Button and then the Step Testing button")
                Original_Test_button = pangolin.VarBool('ui.Original Test', value=False, toggle=False)
                Step_Testing_button = pangolin.VarBool('ui.Step Test', value=False, toggle=False)
                Matplotlib_button = pangolin.VarBool('ui.Matplotlib Save Best', value=False, toggle=False)
                New_Episode_button = pangolin.VarBool('ui.New Episode', value=False, toggle=False)
                Show_Best_button = pangolin.VarBool('ui.Show Best', value=False, toggle=False)
                Translational_Solver_button = pangolin.VarBool('ui.Translation Solver', value=False, toggle=False)
                global_step_counter = 0
                global_epsiode_counter = 0

                bound_box = [[-global_boundx, -global_boundy, 0], [-global_boundx, global_boundy, 0],
                             [global_boundx, global_boundy, 0], [global_boundx, -global_boundy, 0],
                             [-global_boundx, -global_boundy, 0]]
                sac_trainer.load_best_model(path,device)
                # sac_trainer.load_model(path)
                while not pangolin.ShouldQuit():
                    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
                    dcam.Activate(scam)
                    if pangolin.Pushed(Matplotlib_button):
                        fig, ax = plt.subplots()
                        for i in range(env.GT_initials.size()):
                            plotSE2_matplot(env.GT_initials.at(key('x', i )), vehicle_color='g')
                        for i in range(env.Simulated_initials.size()):
                            plotSE2_matplot(env.best_initials.at(key('x', i )), vehicle_color='r')

                        plot2DPoseGraphResult(ax, env.Simulated_graph, env.best_initials, 'r', linewidth=1)
                        plot2DPoseGraphResult(ax, env.GT_graph, env.GT_initials, 'g', linewidth=1)

                        ax.set_xlabel('X Position (m)')
                        ax.set_ylabel('Y Position (m)')
                        ax.set_title('Initial Odometry Estimate (Green) vs RL Estimate (Red)')
                        plt.show()
                    if pangolin.Pushed(New_Episode_button):
                        if global_epsiode_counter == max_evaluation_episodes:
                            print("Max Evaluation Episodes Have Been Reached!")
                        else:

                            global_step_counter = 0
                            env.env_current_step = 0

                            observation = env.reset_rot()
                            # print("Initial graph error squared norm weighted cost is " + str(
                            #     env.Simulated_graph.errorSquaredNorm(env.Simulated_initials) / 2))
                            print("Initial graph error squared norm weighted cost is " + str(
                                env.Simulated_graph.errorSquaredNorm(env.Simulated_initials)))
                            if sqrt_loss:
                                print(
                                    "Initial Non weighted Rotational Graph error SQRT  " + str(
                                        env.initial_cost_rot))
                            else:
                                print(
                                "Initial Non weighted Rotational Graph error squared sum  " + str(env.initial_cost_rot))

                            Simulated_poses, Simulated_btw_factors = updated_trajectory_plotter_for_anchor_zero_idx(
                                env.Simulated_initials, env.Simulated_graph)
                            GT_poses, GT_btw_factors = updated_trajectory_plotter_for_anchor_zero_idx(env.GT_initials,
                                                                                                      env.GT_graph)
                            global_Simulated_poses = Simulated_poses
                            global_Simulated_btw_factors = Simulated_btw_factors
                            global_GT_poses = GT_poses
                            global_GT_btw_factors = GT_btw_factors

                            last_action = action_range * (env.rot_action_space.sample())
                            episode_reward = 0
                            hidden_out = (torch.zeros([1, 1, hidden_dim], dtype=torch.float).to(device),
                                          torch.zeros([1, 1, hidden_dim],
                                                      dtype=torch.float).to(device))  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
                            global_epsiode_counter = global_epsiode_counter + 1

                    if pangolin.Pushed(Step_Testing_button):
                        if global_step_counter == max_steps:
                            print("Max Steps Have Been Reached! Press New Episode To Restart")
                            episode_num = global_epsiode_counter - 1
                            if sqrt_loss:
                                print('Episode: ', episode_num, '| Episode Reward: ', episode_reward,
                                      '| Best Rotation Cost SQRT: ',
                                      env.best_cost_rot)
                            else:
                                print('Episode: ', episode_num, '| Episode Reward: ', episode_reward,
                                  '| Best Rotation Cost squared: ',
                                  env.best_cost_rot)
                        else:
                            residual_ = env.Simulated_graph[env.factor_graph_list_index[(env.env_current_step) % (env.num_btw_fct)]].error(env.Simulated_initials)[2]
                            state = np.append((sac_trainer.RotConv(observation).squeeze(1).cpu().detach().numpy()),residual_)
                            env.env_current_step = env.env_current_step + 1
                            hidden_in = hidden_out
                            action, hidden_out = sac_trainer.policy_net.get_action(state, last_action, hidden_in,device,
                                                                                   deterministic=True)
                            next_observation, reward, done = env.rot_step(action, False)
                            Simulated_poses, Simulated_btw_factors = updated_trajectory_plotter_for_anchor_zero_idx(
                                env.Simulated_initials, env.Simulated_graph)
                            GT_poses, GT_btw_factors = updated_trajectory_plotter_for_anchor_zero_idx(env.GT_initials,
                                                                                                      env.GT_graph)
                            global_Simulated_poses = Simulated_poses
                            global_Simulated_btw_factors = Simulated_btw_factors
                            global_GT_poses = GT_poses
                            global_GT_btw_factors = GT_btw_factors

                            last_action = action
                            episode_reward += reward
                            observation = next_observation
                            global_step_counter = global_step_counter + 1
                            if sqrt_loss:
                                print('Step: ', global_step_counter, '| Non Weighted Rotation Current Cost SQRT: ',
                                      env.current_cost_rot, '| Reward: ', reward)
                            else:
                                print('Step: ', global_step_counter, '| Non Weighted Rotation Current Cost squared: ',
                                  env.current_cost_rot, '| Reward: ', reward)

                    if pangolin.Pushed(Show_Best_button):
                        Simulated_poses, Simulated_btw_factors = updated_trajectory_plotter_for_anchor_zero_idx(
                            env.best_initials, env.Simulated_graph)
                        GT_poses, GT_btw_factors = updated_trajectory_plotter_for_anchor_zero_idx(env.GT_initials,
                                                                                                  env.GT_graph)
                        global_Simulated_poses = Simulated_poses
                        global_Simulated_btw_factors = Simulated_btw_factors
                        global_GT_poses = GT_poses
                        global_GT_btw_factors = GT_btw_factors
                        if sqrt_loss:
                            print('Best SQRT Rotation Cost: ', env.best_cost_rot)
                        else:
                            print('Best Sum Squared Rotation Cost: ', env.best_cost_rot)
                        # print('Total weighted graph error square norm at the optimal rotation pose configuration: ',
                        #       (env.Simulated_graph.get_all_errors(env.best_initials)) / 2)
                        print('Total weighted graph error square norm at the optimal rotation pose configuration: ',
                              (env.Simulated_graph.get_all_errors(env.best_initials)))
                    if pangolin.Pushed(Translational_Solver_button):
                        Trans_2D_Solver(env.Simulated_graph, env.best_initials)
                        Simulated_poses, Simulated_btw_factors = updated_trajectory_plotter_for_anchor_zero_idx(
                            env.best_initials, env.Simulated_graph)
                        GT_poses, GT_btw_factors = updated_trajectory_plotter_for_anchor_zero_idx(env.GT_initials,
                                                                                                  env.GT_graph)
                        global_Simulated_poses = Simulated_poses
                        global_Simulated_btw_factors = Simulated_btw_factors
                        global_GT_poses = GT_poses
                        global_GT_btw_factors = GT_btw_factors

                        # print('Total weighted graph error square norm: ',
                        #       (env.Simulated_graph.get_all_errors(env.best_initials)) / 2)
                        print('Total weighted graph error square norm: ',
                               (env.Simulated_graph.get_all_errors(env.best_initials)) )
                    ############################################################################################################################
                    if pangolin.Pushed(Original_Test_button):

                        sac_trainer.load_best_model(path,device)
                        episode_reward_array = []
                        best_cost_rot_array = []
                        best_total_graph_cost_array = []
                        for eps in range(max_evaluation_episodes):
                            if pangolin.ShouldQuit():
                                break

                            env.env_current_step = 0
                            observation = env.reset_rot()

                            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
                            dcam.Activate(scam)
                            Simulated_poses, Simulated_btw_factors = updated_trajectory_plotter_for_anchor_zero_idx(
                                env.Simulated_initials, env.Simulated_graph)
                            GT_poses, GT_btw_factors = updated_trajectory_plotter_for_anchor_zero_idx(env.GT_initials,
                                                                                                      env.GT_graph)
                            global_Simulated_poses = Simulated_poses
                            global_Simulated_btw_factors = Simulated_btw_factors
                            global_GT_poses = GT_poses
                            global_GT_btw_factors = GT_btw_factors
                            gl.glLineWidth(1.5)
                            gl.glColor3f(1.0, 1.0, 1.0)
                            pangolin.DrawLine(bound_box)
                            gl.glLineWidth(1)

                            gl.glColor3f(1, 0.0, 0.0)
                            for i in range(len(global_Simulated_poses)):
                                pangolin.DrawLine(global_Simulated_poses[i], point_size=0)
                            for i in range(len(global_Simulated_btw_factors)):
                                pangolin.DrawLine(global_Simulated_btw_factors[i], point_size=0)
                            gl.glColor3f(0.0, 1.0, 0.0)
                            for i in range(len(global_GT_poses)):
                                pangolin.DrawLine(global_GT_poses[i], point_size=0)
                            for i in range(len(global_GT_btw_factors)):
                                pangolin.DrawLine(global_GT_btw_factors[i], point_size=0)
                            pangolin.FinishFrame()

                            last_action = action_range * (env.rot_action_space.sample())
                            episode_reward = 0
                            hidden_out = (torch.zeros([1, 1, hidden_dim], dtype=torch.float).to(device),
                                          torch.zeros([1, 1, hidden_dim],
                                                      dtype=torch.float).to(device))  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)

                            for step in range(max_steps):
                                if pangolin.ShouldQuit():
                                    break
                                gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
                                dcam.Activate(scam)
                                residual_ = env.Simulated_graph[env.factor_graph_list_index[(env.env_current_step) % (env.num_btw_fct)]].error(env.Simulated_initials)[2]
                                state = np.append((sac_trainer.RotConv(observation).squeeze(1).cpu().detach().numpy()),residual_)
                                env.env_current_step = env.env_current_step + 1
                                hidden_in = hidden_out
                                action, hidden_out = sac_trainer.policy_net.get_action(state, last_action, hidden_in,device,
                                                                                       deterministic=True)
                                next_observation, reward, done = env.rot_step(action, False)
                                if (step >= max_steps - (max_steps/args.max_steps)) or (step <=  (max_steps/args.max_steps)) :
                                    if step == max_steps - 1:
                                        Trans_2D_Solver(env.Simulated_graph, env.best_initials)
                                        Simulated_poses, Simulated_btw_factors = updated_trajectory_plotter_for_anchor_zero_idx(
                                            env.best_initials, env.Simulated_graph)
                                        GT_poses, GT_btw_factors = updated_trajectory_plotter_for_anchor_zero_idx(
                                            env.GT_initials, env.GT_graph)
                                        global_Simulated_poses = Simulated_poses
                                        global_Simulated_btw_factors = Simulated_btw_factors
                                        global_GT_poses = GT_poses
                                        global_GT_btw_factors = GT_btw_factors

                                        gl.glLineWidth(1.5)
                                        gl.glColor3f(1.0, 1.0, 1.0)
                                        pangolin.DrawLine(bound_box)
                                        gl.glLineWidth(1)

                                        gl.glColor3f(1, 0.0, 0.0)
                                        for i in range(len(global_Simulated_poses)):
                                            pangolin.DrawLine(global_Simulated_poses[i], point_size=0)
                                        for i in range(len(global_Simulated_btw_factors)):
                                            pangolin.DrawLine(global_Simulated_btw_factors[i], point_size=0)
                                        gl.glColor3f(0.0, 1.0, 0.0)
                                        for i in range(len(global_GT_poses)):
                                            pangolin.DrawLine(global_GT_poses[i], point_size=0)
                                        for i in range(len(global_GT_btw_factors)):
                                            pangolin.DrawLine(global_GT_btw_factors[i], point_size=0)
                                        pangolin.FinishFrame()
                                    else:
                                        Simulated_poses, Simulated_btw_factors = updated_trajectory_plotter_for_anchor_zero_idx(
                                            env.Simulated_initials, env.Simulated_graph)
                                        GT_poses, GT_btw_factors = updated_trajectory_plotter_for_anchor_zero_idx(
                                            env.GT_initials, env.GT_graph)
                                        global_Simulated_poses = Simulated_poses
                                        global_Simulated_btw_factors = Simulated_btw_factors
                                        global_GT_poses = GT_poses
                                        global_GT_btw_factors = GT_btw_factors

                                        gl.glLineWidth(1.5)
                                        gl.glColor3f(1.0, 1.0, 1.0)
                                        pangolin.DrawLine(bound_box)
                                        gl.glLineWidth(1)

                                        gl.glColor3f(1, 0.0, 0.0)
                                        for i in range(len(global_Simulated_poses)):
                                            pangolin.DrawLine(global_Simulated_poses[i], point_size=0)
                                        for i in range(len(global_Simulated_btw_factors)):
                                            pangolin.DrawLine(global_Simulated_btw_factors[i], point_size=0)
                                        gl.glColor3f(0.0, 1.0, 0.0)
                                        for i in range(len(global_GT_poses)):
                                            pangolin.DrawLine(global_GT_poses[i], point_size=0)
                                        for i in range(len(global_GT_btw_factors)):
                                            pangolin.DrawLine(global_GT_btw_factors[i], point_size=0)
                                        pangolin.FinishFrame()

                                last_action = action
                                episode_reward += reward
                                observation = next_observation


                            # graph_total_cost = ((env.Simulated_graph.get_all_errors(env.best_initials)) / 2)
                            graph_total_cost = ((env.Simulated_graph.get_all_errors(env.best_initials)))
                            episode_reward_array.append(episode_reward)
                            best_cost_rot_array.append(env.best_cost_rot)
                            best_total_graph_cost_array.append(graph_total_cost)
                            list1 = np.array(episode_reward_array)
                            list2 = np.array(best_cost_rot_array)
                            list3 = np.array(best_total_graph_cost_array)
                            if sqrt_loss:
                                print('Episode: ', eps, '| Episode Reward: ', episode_reward, '| Best Rotation Cost SQRT: ',
                                      env.best_cost_rot, '| Best Total Graph Cost(weighted error square norm): ',
                                      graph_total_cost)
                            else:
                                print('Episode: ', eps, '| Episode Reward: ', episode_reward, '| Best Rotation Cost Squared: ',
                                  env.best_cost_rot, '| Best Total Graph Cost(weighted error square norm): ',
                                  graph_total_cost)
                        print("Average Reward in " + str(max_evaluation_episodes) + " episodes", np.mean(list1))
                        if sqrt_loss:
                            print("Average Best Rotation SQRT loss(non weighted) in " + str(
                                max_evaluation_episodes) + " episodes", np.mean(list2))
                        else:
                            print("Average Best Rotation sum squared cost (non weighted) in " + str(
                                max_evaluation_episodes) + " episodes", np.mean(list2))


                        print(
                            "Average Best Graph error squared norm (weighted) in " + str(
                                max_evaluation_episodes) + " episodes", np.mean(list3))

                    gl.glLineWidth(1.5)
                    gl.glColor3f(1.0, 1.0, 1.0)
                    pangolin.DrawLine(bound_box)
                    gl.glLineWidth(1)
                    #original
                    gl.glColor3f(1.0, 0.0, 0.0)
                    #gl.glColor3f(0.0, 1.0, 0.0)
                    for i in range(len(global_Simulated_poses)):
                        pangolin.DrawLine(global_Simulated_poses[i], point_size=0)
                    for i in range(len(global_Simulated_btw_factors)):
                        pangolin.DrawLine(global_Simulated_btw_factors[i], point_size=0)
                    gl.glColor3f(0.0, 1.0, 0.0)
                    for i in range(len(global_GT_poses)):
                        pangolin.DrawLine(global_GT_poses[i], point_size=0)
                    for i in range(len(global_GT_btw_factors)):
                        pangolin.DrawLine(global_GT_btw_factors[i], point_size=0)
                    pangolin.FinishFrame()
        else:
            Best_Initials_to_write = Variables()
            if args.change:
                if args.load:
                    env = Load_Pose_Graph(global_boundx, global_boundy, path,sqrt_loss,state_dim,frob,reward_numerator_scale,r_act_range)
                    action_space = env.rot_action_space
                    state_space = env.rot_observation_space
                    action_dim = action_space.shape[0]
                    action_range = args.rot_action_range
                    max_steps = args.max_steps * env.num_btw_fct
                    frequency = args.freq_of_change
                else:
                    env = PoseGraph(global_boundx, global_boundy, number_of_nodes, path,trans_noise_,rot_noise_,inter_dist,sqrt_loss,change_noise,loop_close,prob_of_loop,state_dim,frob,reward_numerator_scale,r_act_range)
                    max_steps = args.max_steps * env.num_btw_fct
                    # action_space = env.action_space
                    action_space = env.rot_action_space
                    state_space = env.rot_observation_space
                    action_dim = action_space.shape[0]
                    # action_range = return_action_range(env.num_actions, 5, 1)
                    frequency = args.freq_of_change
                    action_range = args.rot_action_range

                sac_trainer = SAC_Trainer_rotation_w_grad_option_GCN(replay_buffer, state_space, action_space, lr,
                                                                 grad, grad_value, hidden_dim=hidden_dim,
                                                                 action_range=action_range)
                best_graph_cost = 1e50
                sac_trainer.load_best_model(path,device)
                #sac_trainer.load_model(path)
                episode_reward_array = []
                best_cost_rot_array = []
                time_array= []
                best_total_graph_cost_array = []
                for eps in range(max_evaluation_episodes):
                    if eps % frequency == 0:
                        env.change_target(False, False)
                    env.env_current_step = 0
                    observation = env.reset_rot()
                    last_action = action_range * (env.rot_action_space.sample())
                    episode_reward = 0
                    hidden_out = (torch.zeros([1, 1, hidden_dim], dtype=torch.float).to(device),
                                  torch.zeros([1, 1, hidden_dim],
                                              dtype=torch.float).to(device))  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
                    t0 = time.time()
                    for step in range(max_steps):
                        residual_ = env.Simulated_graph[env.factor_graph_list_index[(env.env_current_step) % (env.num_btw_fct)]].error(env.Simulated_initials)[2]
                        state = np.append((sac_trainer.RotConv(observation).squeeze(1).cpu().detach().numpy()),residual_)
                        env.env_current_step = env.env_current_step + 1
                        hidden_in = hidden_out
                        action, hidden_out = sac_trainer.policy_net.get_action(state, last_action, hidden_in,device,
                                                                               deterministic=True)
                        next_observation, reward, done = env.rot_step(action, False)


                        last_action = action
                        episode_reward += reward
                        observation = next_observation
                    Trans_2D_Solver(env.Simulated_graph, env.best_initials)
                    t1 = time.time()
                    diff = t1-t0
                    time_array.append(diff)


                    # graph_total_cost = ((env.Simulated_graph.get_all_errors(env.best_initials)) / 2)
                    graph_total_cost = ((env.Simulated_graph.get_all_errors(env.best_initials)))
                    if graph_total_cost<best_graph_cost:
                        best_graph_cost = graph_total_cost
                        Best_Initials_to_write = env.best_initials
                    episode_reward_array.append(episode_reward)
                    best_cost_rot_array.append(env.best_cost_rot)
                    best_total_graph_cost_array.append(graph_total_cost)
                    list1 = np.array(episode_reward_array)
                    list2 = np.array(best_cost_rot_array)
                    list3 = np.array(best_total_graph_cost_array)
                    list4 = np.array(time_array)

                    if sqrt_loss:
                        print('Episode: ', eps, '| Episode Reward: ', episode_reward, '| Initial Rot Cost: ',
                              env.initial_cost_rot, '| Best Rotation Cost SQRT: ',
                              env.best_cost_rot, '| Best Total Graph Cost(weighted error square norm): ',
                              graph_total_cost)
                    else:
                        print('Episode: ', eps, '| Episode Reward: ', episode_reward,'| Initial Rot Cost: ', env.initial_cost_rot,'| Best Rotation Cost Squared: ',
                          env.best_cost_rot, '| Best Total Graph Cost(weighted error square norm): ',
                          graph_total_cost)
                print("Average Reward in " + str(max_evaluation_episodes) + " episodes", np.mean(list1))
                if sqrt_loss:
                    print("Average Best Rotation SQRT loss(non weighted) in " + str(
                        max_evaluation_episodes) + " episodes", np.mean(list2))
                else:
                    print("Average Best Rotation sum squared cost (non weighted) in " + str(
                    max_evaluation_episodes) + " episodes", np.mean(list2))
                print(
                    "Average Best Graph error squared norm (weighted) in " + str(
                        max_evaluation_episodes) + " episodes", np.mean(list3))
                print("Average time per episode " , np.mean(list4))
                writeG2O(env.path+"/Simulated_graphs/Simulated_graph_best_from"+ str(max_evaluation_episodes)+".g2o", env.Simulated_graph, Best_Initials_to_write)


            else:
                if args.load:
                    env = Load_Pose_Graph(global_boundx, global_boundy, path,sqrt_loss,state_dim,frob,reward_numerator_scale,r_act_range)
                    action_space = env.rot_action_space
                    state_space = env.rot_observation_space
                    action_dim = action_space.shape[0]
                    action_range = args.rot_action_range
                    max_steps = args.max_steps * env.num_btw_fct
                    frequency = args.freq_of_change
                else:
                    env = PoseGraph(global_boundx, global_boundy, number_of_nodes, path,trans_noise_,rot_noise_,inter_dist,sqrt_loss,change_noise,loop_close,prob_of_loop,state_dim,frob,reward_numerator_scale,r_act_range)
                    max_steps = args.max_steps * env.num_btw_fct
                    # action_space = env.action_space
                    action_space = env.rot_action_space
                    state_space = env.rot_observation_space
                    action_dim = action_space.shape[0]
                    # action_range = return_action_range(env.num_actions, 5, 1)
                    frequency = args.freq_of_change
                    action_range = args.rot_action_range

                sac_trainer = SAC_Trainer_rotation_w_grad_option_GCN(replay_buffer, state_space, action_space, lr,
                                                                 grad, grad_value, hidden_dim=hidden_dim,
                                                                 action_range=action_range)
                best_graph_cost = 1e50
                sac_trainer.load_best_model(path,device)
                # sac_trainer.load_model(path)
                episode_reward_array = []
                best_cost_rot_array = []
                time_array = []
                best_total_graph_cost_array = []
                for eps in range(max_evaluation_episodes):

                    env.env_current_step = 0
                    observation = env.reset_rot()
                    last_action = action_range * (env.rot_action_space.sample())
                    episode_reward = 0
                    hidden_out = (torch.zeros([1, 1, hidden_dim], dtype=torch.float).to(device),
                                  torch.zeros([1, 1, hidden_dim],
                                              dtype=torch.float).to(device))  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
                    t0 = time.time()
                    for step in range(max_steps):
                        residual_ = env.Simulated_graph[env.factor_graph_list_index[(env.env_current_step) % (env.num_btw_fct)]].error(env.Simulated_initials)[2]
                        state = np.append((sac_trainer.RotConv(observation).squeeze(1).cpu().detach().numpy()),residual_)
                        env.env_current_step = env.env_current_step + 1
                        hidden_in = hidden_out
                        action, hidden_out = sac_trainer.policy_net.get_action(state, last_action, hidden_in,device,
                                                                               deterministic=True)
                        next_observation, reward, done = env.rot_step(action, False)

                        last_action = action
                        episode_reward += reward
                        observation = next_observation
                    Trans_2D_Solver(env.Simulated_graph, env.best_initials)
                    t1 = time.time()
                    diff = t1-t0
                    time_array.append(diff)
                    print("Printing_all_errors " +str(eps))
                    env.Simulated_graph.print_all_errors(env.best_initials)

                    # graph_total_cost = ((env.Simulated_graph.get_all_errors(env.best_initials)) / 2)
                    graph_total_cost = ((env.Simulated_graph.get_all_errors(env.best_initials)))
                    if graph_total_cost < best_graph_cost:
                        best_graph_cost = graph_total_cost
                        Best_Initials_to_write = env.best_initials
                    episode_reward_array.append(episode_reward)
                    best_cost_rot_array.append(env.best_cost_rot)
                    best_total_graph_cost_array.append(graph_total_cost)
                    list1 = np.array(episode_reward_array)
                    list2 = np.array(best_cost_rot_array)
                    list3 = np.array(best_total_graph_cost_array)
                    list4 = np.array(time_array)
                    if sqrt_loss:
                        print('Episode: ', eps, '| Episode Reward: ', episode_reward, '| Initial Rot Cost: ',
                              env.initial_cost_rot, '| Best Rotation Cost SQRT: ',
                              env.best_cost_rot, '| Best Total Graph Cost(weighted error square norm): ',
                              graph_total_cost)
                    else:
                        print('Episode: ', eps, '| Episode Reward: ', episode_reward, '| Initial Rot Cost: ',
                          env.initial_cost_rot, '| Best Rotation Cost Sqaured: ',
                          env.best_cost_rot, '| Best Total Graph Cost(weighted error square norm): ',
                          graph_total_cost)
                print("Average Reward in " + str(max_evaluation_episodes) + " episodes", np.mean(list1))
                if sqrt_loss:
                    print("Average Best Rotation SQRT loss(non weighted) in " + str(
                        max_evaluation_episodes) + " episodes", np.mean(list2))
                else:
                    print("Average Best Rotation sum squared cost (non weighted) in " + str(
                    max_evaluation_episodes) + " episodes", np.mean(list2))

                print(
                    "Average Best Graph error squared norm (weighted) in " + str(
                        max_evaluation_episodes) + " episodes", np.mean(list3))
                print("Average time per episode ", np.mean(list4))
                writeG2O(
                    env.path + "/Simulated_graphs/Simulated_graph_best_from" + str(max_evaluation_episodes) + ".g2o",
                    env.Simulated_graph, Best_Initials_to_write)
