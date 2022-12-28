import pandas as pd
import os
import torch.optim as optim
from common.value_networks import *
from common.policy_networks import *
import dgl
from dgl.base import DGLError
import dgl.function as fn
import torch as th
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.lines import Line2D
from scipy.ndimage.filters import gaussian_filter1d
import itertools




# #GPU = True
# GPU = False
# device_idx = 0
# if GPU:
#     device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
# else:
#     device = torch.device("cpu")
device_idx = 0
device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")


class RotConv(nn.Module):
    ### between factors can be added in any order loop first or odometry first but anchor should be idx = 0
    def __init__(self, out_feat):
        super(RotConv, self).__init__()

        self.beta = nn.Linear(1, 1)
        self.linear1 = nn.Linear(1,out_feat)

    def forward(self, g):
        ### Takes in g which is either a batch of homogeneous graph or a single homogeneous graph and returns cost which is (batch size,1) if single graph will return (1,1)
        if (g.in_degrees() == 0).any():
            raise DGLError('There are 0-in-degree nodes in the graph, '
                           'output for those nodes will be invalid. '
                           'This is harmful for some applications, '
                           'causing silent performance regression. '
                           'Adding self-loop on the input graph by '
                           'calling `g = dgl.add_self_loop(g)` will resolve '
                           'the issue. Setting ``allow_zero_in_degree`` '
                           'to be `True` when constructing this module will '
                           'suppress the check and let the code run.')
        #TODO
        # below is with info weighting and *0.5
        # g.apply_edges(lambda edges: {'tmp_cost': (((th.linalg.matrix_norm(
        #     th.matmul(edges.src['rot_v'], edges.data['rot_meas']) - edges.dst['rot_v'], ord='fro')) * edges.data[
        #                                                      'sq_info']) * 0.5).unsqueeze(dim=1)})

        g.apply_edges(lambda edges: {'tmp_cost': (th.linalg.matrix_norm(
            th.matmul(edges.src['rot_v'], edges.data['rot_meas']) - edges.dst['rot_v'], ord='fro')).unsqueeze(dim=1)})

        g.edata['tmp_cost'] = th.sigmoid(self.beta(g.edata['tmp_cost'].float()))
        g.update_all(fn.copy_e('tmp_cost', 'm'), fn.sum('m', 'cost'))


        return self.linear1(dgl.mean_nodes(g, 'cost'))


class Original_SAC_Trainer():
    def __init__(self, state_dim,action_dim,replay_buffer, hidden_dim, action_range):
        self.replay_buffer = replay_buffer

        self.soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range).to(device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)
        print('Soft Q Network (1,2): ', self.soft_q_net1)
        print('Policy Network: ', self.policy_net)

        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        soft_q_lr = 3e-4
        policy_lr = 3e-4
        alpha_lr = 3e-4

        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    def update(self, batch_size, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99, soft_tau=1e-2):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(
            device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2 = self.soft_q_net2(state, action)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(
            dim=0) + 1e-6)  # normalize with batch mean and std; plus a small number to prevent numerical problem
        # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q)
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            # print('alpha loss: ',alpha_loss)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 1.
            alpha_loss = 0

        # Training Q Function
        target_q_min = torch.min(self.target_soft_q_net1(next_state, new_next_action),
                                 self.target_soft_q_net2(next_state, new_next_action)) - self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * gamma * target_q_min  # if done==1, only reward
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1.float(),
                                               target_q_value.float().detach())  # detach: no gradients for the variable
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2.float(), target_q_value.float().detach())

        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()

        # Training Policy Function
        predicted_new_q_value = torch.min(self.soft_q_net1(state, new_action), self.soft_q_net2(state, new_action))
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # print('q loss: ', q_value_loss1, q_value_loss2)
        # print('policy loss: ', policy_loss )

        # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        return q_value_loss1.detach().cpu(), q_value_loss2.detach().cpu(), policy_loss.detach().cpu(), alpha_loss.detach().cpu()

    def save_checkpoint(self, path, rewards):
        try:
            os.makedirs(path)
        except:
            pass

        torch.save(self.soft_q_net1.state_dict(), path + '/_q1_rot')
        torch.save(self.soft_q_net2.state_dict(), path + '/_q2_rot')
        torch.save(self.policy_net.state_dict(), path + '/_policy_rot')

        plt.cla()
        plt.plot(rewards, c='#bd0e3a', alpha=0.3)
        plt.plot(gaussian_filter1d(rewards, sigma=5), c='#bd0e3a', label='Rewards')
        plt.xlabel('Episodes')
        plt.ylabel('Cumulative reward')
        plt.title('Sac_Rotation')
        plt.savefig(os.path.join(path, 'reward_rotation.png'))

        pd.DataFrame(rewards, columns=['Reward']).to_csv(os.path.join(path, 'rewards_rotation.csv'), index=False)

    def save_model(self, path):
        torch.save(self.soft_q_net1.state_dict(), path + '/_q1_rot')
        torch.save(self.soft_q_net2.state_dict(), path + '/_q2_rot')
        torch.save(self.policy_net.state_dict(), path + '/_policy_rot')

    def save_best_model(self, path):
        torch.save(self.soft_q_net1.state_dict(), path + '/_q1_best_rot')
        torch.save(self.soft_q_net2.state_dict(), path + '/_q2_best_rot')
        torch.save(self.policy_net.state_dict(), path + '/_policy_best_rot')

    def load_model(self, path):
        self.soft_q_net1.load_state_dict(torch.load(path + '/_q1_rot'))
        self.soft_q_net2.load_state_dict(torch.load(path + '/_q2_rot'))
        self.policy_net.load_state_dict(torch.load(path + '/_policy_rot'))

        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()

    def load_best_model(self, path):
        self.soft_q_net1.load_state_dict(torch.load(path + '/_q1_best_rot'))
        self.soft_q_net2.load_state_dict(torch.load(path + '/_q2_best_rot'))
        self.policy_net.load_state_dict(torch.load(path + '/_policy_best_rot'))

        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()



class SAC_Trainer_rotation_w_grad_option_GCN_all_actions():
    def __init__(self, replay_buffer, state_space, action_space, lr, grad, grad_value, hidden_dim, action_range):
        self.replay_buffer = replay_buffer
        self.lr = lr
        self.grad = grad
        self.grad_value = grad_value
        self.state_dimension=(state_space.shape[0])
        self.RotConv = RotConv(out_feat=(state_space.shape[0])).to(device)
        self.soft_q_net1 = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.soft_q_net2 = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.target_soft_q_net1 = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.target_soft_q_net2 = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.target_soft_q_net1.eval().requires_grad_(False)
        self.target_soft_q_net2.eval().requires_grad_(False)
        self.policy_net = SAC_PolicyNetworkLSTM_rotation(state_space, action_space, hidden_dim, action_range).to(device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)
        print('Soft Q Network (1,2): ', self.soft_q_net1)
        print('Policy Network: ', self.policy_net)

        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        soft_q_lr = self.lr
        policy_lr = self.lr
        alpha_lr = self.lr
        self.actor_loss_weight = policy_lr / soft_q_lr
        # Original lr
        # soft_q_lr = 3e-4
        # policy_lr = 3e-4
        # alpha_lr = 3e-4

        # soft_q_lr = 3e-5
        # policy_lr = 3e-5
        # alpha_lr = 3e-5

        # soft_q_lr = 3e-6
        # policy_lr = 3e-6
        # alpha_lr = 3e-6

        self.optimizer = optim.Adam(itertools.chain(self.RotConv.parameters(),
                                                    self.soft_q_net1.parameters(),
                                                    self.soft_q_net2.parameters(),
                                                    self.policy_net.parameters()),lr=soft_q_lr)

        # self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=soft_q_lr)
        # self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=soft_q_lr)
        # self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        # Original reward_scale = 10

    def update(self, path, batch_size, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99,
               soft_tau=1e-2):
        hidden_in, hidden_out, observation, action, last_action, reward, next_observation, done= self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)
        observation = dgl.batch(observation)
        next_observation = dgl.batch(next_observation)

        #state=torch.FloatTensor((self.RotConv(observation)).reshape(batch_size,-1,self.state_dimension)).to(device)
        state = (self.RotConv(observation)).reshape(batch_size, -1, self.state_dimension)
        with torch.no_grad():
            #next_state = torch.FloatTensor((self.RotConv(next_observation)).reshape(batch_size, -1, self.state_dimension)).to(device)
            next_state = (self.RotConv(next_observation)).reshape(batch_size, -1, self.state_dimension)

        action = torch.FloatTensor(action).to(device)
        last_action = torch.FloatTensor(last_action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(-1).to(
            device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done = torch.FloatTensor(np.float32(done)).unsqueeze(-1).to(device)
        # normalize with batch mean and std; plus a small number to prevent numerical problem
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(
            dim=0) + 1e-6)

        # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q)
        new_action, log_prob, z, mean, log_std, _ = self.policy_net.evaluate(state, last_action, hidden_in)
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            # print('alpha loss: ',alpha_loss)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            with torch.no_grad():
                self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 1.
            alpha_loss = 0


        predicted_q_value1, _ = self.soft_q_net1(state, action, last_action, hidden_in)
        predicted_q_value2, _ = self.soft_q_net2(state, action, last_action, hidden_in)
        with torch.no_grad():
            new_next_action, next_log_prob, _, _, _, _ = self.policy_net.evaluate(next_state, action, hidden_out)
            # Training Q Function
            predict_target_q1, _ = self.target_soft_q_net1(next_state, new_next_action, action, hidden_out)
            predict_target_q2, _ = self.target_soft_q_net2(next_state, new_next_action, action, hidden_out)
            target_q_min = torch.min(predict_target_q1, predict_target_q2) - self.alpha * next_log_prob
            target_q_value = reward + (1 - done) * gamma * target_q_min  # if done==1, only reward
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1.float(),
                                               target_q_value.float())  # detach: no gradients for the variable
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2.float(), target_q_value.float())
        q_loss = (q_value_loss1+q_value_loss2)/2.0

        # Training Policy Function
        predict_q1, _ = self.soft_q_net1(state, new_action, last_action, hidden_in)
        predict_q2, _ = self.soft_q_net2(state, new_action, last_action, hidden_in)
        predicted_new_q_value = torch.min(predict_q1, predict_q2)

        predict_q1_critic_grad_only, _ = self.soft_q_net1(state, new_action.detach(), last_action.detach(), hidden_in)
        predict_q2_critic_grad_only, _ = self.soft_q_net2(state, new_action.detach(), last_action.detach(), hidden_in)
        predicted_new_q_value_critic_grad_only = torch.min(predict_q1_critic_grad_only, predict_q2_critic_grad_only)

        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()
        policy_loss_unbiased = policy_loss+ predicted_new_q_value_critic_grad_only.mean()
        loss = q_loss+self.actor_loss_weight*policy_loss_unbiased
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad == True:
            for param_group in self.optimizer.param_groups:
                nn.utils.clip_grad.clip_grad_norm_(param_group['params'],
                                                   max_norm=self.grad_value,
                                                   norm_type=2)

        self.optimizer.step()

        # print('q loss: ', q_value_loss1, q_value_loss2)
        # print('policy loss: ', policy_loss )

        # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        return q_loss.detach().cpu(), policy_loss.detach().cpu(), alpha_loss.detach().cpu(),loss.detach().cpu()

    def save_checkpoint(self, path, rewards):
        try:
            os.makedirs(path)
        except:
            pass

        torch.save(self.RotConv.state_dict(), path + '/_rot_conv_enc')
        torch.save(self.soft_q_net1.state_dict(), path + '/_q1_rot')
        torch.save(self.soft_q_net2.state_dict(), path + '/_q2_rot')
        torch.save(self.policy_net.state_dict(), path + '/_policy_rot')

        plt.cla()
        plt.plot(rewards, c='#bd0e3a', alpha=0.3)
        plt.plot(gaussian_filter1d(rewards, sigma=5), c='#bd0e3a', label='Rewards')
        plt.xlabel('Episodes')
        plt.ylabel('Cumulative reward')
        plt.title('Sac_LSTM_Rotation')
        plt.savefig(os.path.join(path, 'reward_rotation.png'))

        pd.DataFrame(rewards, columns=['Reward']).to_csv(os.path.join(path, 'rewards_rotation.csv'), index=False)

    def save_model(self, path):
        torch.save(self.RotConv.state_dict(), path + '/_rot_conv_enc')
        torch.save(self.soft_q_net1.state_dict(), path + '/_q1_rot')
        torch.save(self.soft_q_net2.state_dict(), path + '/_q2_rot')
        torch.save(self.policy_net.state_dict(), path + '/_policy_rot')

    def save_best_model(self, path):
        torch.save(self.RotConv.state_dict(), path + '/_rot_best_conv_enc')
        torch.save(self.soft_q_net1.state_dict(), path + '/_q1_best_rot')
        torch.save(self.soft_q_net2.state_dict(), path + '/_q2_best_rot')
        torch.save(self.policy_net.state_dict(), path + '/_policy_best_rot')

    def load_model(self, path):
        self.RotConv.load_state_dict(torch.load(path + '/_rot_conv_enc'))
        self.soft_q_net1.load_state_dict(torch.load(path + '/_q1_rot'))
        self.soft_q_net2.load_state_dict(torch.load(path + '/_q2_rot'))
        self.policy_net.load_state_dict(torch.load(path + '/_policy_rot'))

        self.RotConv.eval()
        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()

    def load_model_for_grad_flow(self, path):
        self.RotConv.load_state_dict(torch.load(path + '/_rot_conv_enc'))
        self.soft_q_net1.load_state_dict(torch.load(path + '/_q1_rot'))
        self.soft_q_net2.load_state_dict(torch.load(path + '/_q2_rot'))
        self.policy_net.load_state_dict(torch.load(path + '/_policy_rot'))

    def load_best_model(self, path):
        self.RotConv.load_state_dict(torch.load(path + '/_rot_best_conv_enc'))
        self.soft_q_net1.load_state_dict(torch.load(path + '/_q1_best_rot'))
        self.soft_q_net2.load_state_dict(torch.load(path + '/_q2_best_rot'))
        self.policy_net.load_state_dict(torch.load(path + '/_policy_best_rot'))

        self.RotConv.eval()
        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()


class SAC_Trainer_rotation_w_grad_option_GCN():
    def __init__(self, replay_buffer, state_space, action_space, lr, grad, grad_value, hidden_dim, action_range):
        self.replay_buffer = replay_buffer
        self.lr = lr
        self.grad = grad
        self.grad_value = grad_value
        self.state_dimension=(state_space.shape[0])-1
        self.RotConv = RotConv(out_feat=(state_space.shape[0])-1).to(device)
        self.soft_q_net1 = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.soft_q_net2 = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.target_soft_q_net1 = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.target_soft_q_net2 = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.target_soft_q_net1.eval().requires_grad_(False)
        self.target_soft_q_net2.eval().requires_grad_(False)
        self.policy_net = SAC_PolicyNetworkLSTM_rotation(state_space, action_space, hidden_dim, action_range).to(device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)
        print('Soft Q Network (1,2): ', self.soft_q_net1)
        print('Policy Network: ', self.policy_net)

        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        soft_q_lr = self.lr
        policy_lr = self.lr
        alpha_lr = self.lr
        self.actor_loss_weight = policy_lr / soft_q_lr
        # Original lr
        # soft_q_lr = 3e-4
        # policy_lr = 3e-4
        # alpha_lr = 3e-4

        # soft_q_lr = 3e-5
        # policy_lr = 3e-5
        # alpha_lr = 3e-5

        # soft_q_lr = 3e-6
        # policy_lr = 3e-6
        # alpha_lr = 3e-6

        self.optimizer = optim.Adam(itertools.chain(self.RotConv.parameters(),
                                                    self.soft_q_net1.parameters(),
                                                    self.soft_q_net2.parameters(),
                                                    self.policy_net.parameters()),lr=soft_q_lr)

        # self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=soft_q_lr)
        # self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=soft_q_lr)
        # self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        # Original reward_scale = 10

    def update(self, path, batch_size, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99,
               soft_tau=1e-2):
        hidden_in, hidden_out, observation, action, last_action, reward, next_observation, done,residual_at_agent, next_residual_at_agent = self.replay_buffer.sample(
            batch_size)
        # print('sample:', state, action,  reward, done)
        observation = dgl.batch(observation)
        next_observation = dgl.batch(next_observation)
        residual_at_agent = th.FloatTensor(residual_at_agent).to(device)
        next_residual_at_agent = th.FloatTensor(next_residual_at_agent).to(device)
        state=(self.RotConv(observation)).reshape(batch_size,-1,self.state_dimension)
        state = (th.cat((state, residual_at_agent.unsqueeze(2)), dim=2)).to(device)
        with torch.no_grad():
            next_state = (self.RotConv(next_observation)).reshape(batch_size, -1, self.state_dimension)
            next_state = (th.cat((next_state, next_residual_at_agent.unsqueeze(2)), dim=2)).to(device)
        action = torch.FloatTensor(np.array(action)).to(device)
        last_action = torch.FloatTensor(np.array(last_action)).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(-1).to(
            device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done = torch.FloatTensor(np.float32(done)).unsqueeze(-1).to(device)
        # normalize with batch mean and std; plus a small number to prevent numerical problem
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(
            dim=0) + 1e-6)

        # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q)
        new_action, log_prob, z, mean, log_std, _ = self.policy_net.evaluate(state, last_action, hidden_in)
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            # print('alpha loss: ',alpha_loss)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            with torch.no_grad():
                self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 1.
            alpha_loss = 0


        predicted_q_value1, _ = self.soft_q_net1(state, action, last_action, hidden_in)
        predicted_q_value2, _ = self.soft_q_net2(state, action, last_action, hidden_in)
        with torch.no_grad():
            new_next_action, next_log_prob, _, _, _, _ = self.policy_net.evaluate(next_state, action, hidden_out)
            # Training Q Function
            predict_target_q1, _ = self.target_soft_q_net1(next_state, new_next_action, action, hidden_out)
            predict_target_q2, _ = self.target_soft_q_net2(next_state, new_next_action, action, hidden_out)
            target_q_min = torch.min(predict_target_q1, predict_target_q2) - self.alpha * next_log_prob
            target_q_value = reward + (1 - done) * gamma * target_q_min  # if done==1, only reward
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1.float(),
                                               target_q_value.float())  # detach: no gradients for the variable
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2.float(), target_q_value.float())
        q_loss = (q_value_loss1+q_value_loss2)/2.0

        # Training Policy Function
        predict_q1, _ = self.soft_q_net1(state, new_action, last_action, hidden_in)
        predict_q2, _ = self.soft_q_net2(state, new_action, last_action, hidden_in)
        predicted_new_q_value = torch.min(predict_q1, predict_q2)

        predict_q1_critic_grad_only, _ = self.soft_q_net1(state, new_action.detach(), last_action.detach(), hidden_in)
        predict_q2_critic_grad_only, _ = self.soft_q_net2(state, new_action.detach(), last_action.detach(), hidden_in)

        predicted_new_q_value_critic_grad_only = torch.min(predict_q1_critic_grad_only, predict_q2_critic_grad_only)

        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()
        policy_loss_unbiased = policy_loss+ predicted_new_q_value_critic_grad_only.mean()
        loss = q_loss+self.actor_loss_weight*policy_loss_unbiased
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad == True:
            for param_group in self.optimizer.param_groups:
                nn.utils.clip_grad.clip_grad_norm_(param_group['params'],
                                                   max_norm=self.grad_value,
                                                   norm_type=2)

        self.optimizer.step()

        # print('q loss: ', q_value_loss1, q_value_loss2)
        # print('policy loss: ', policy_loss )

        # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        return q_loss.detach().cpu(), policy_loss.detach().cpu(), alpha_loss.detach().cpu(),loss.detach().cpu()

    def save_checkpoint(self, path, rewards):
        try:
            os.makedirs(path)
        except:
            pass

        torch.save(self.RotConv.state_dict(), path + '/_rot_conv_enc')
        torch.save(self.soft_q_net1.state_dict(), path + '/_q1_rot')
        torch.save(self.soft_q_net2.state_dict(), path + '/_q2_rot')
        torch.save(self.policy_net.state_dict(), path + '/_policy_rot')

        plt.cla()
        plt.plot(rewards, c='#bd0e3a', alpha=0.3)
        plt.plot(gaussian_filter1d(rewards, sigma=5), c='#bd0e3a', label='Rewards')
        plt.xlabel('Episodes')
        plt.ylabel('Cumulative reward')
        plt.title('Sac_LSTM_Rotation')
        plt.savefig(os.path.join(path, 'reward_rotation.png'))

        pd.DataFrame(rewards, columns=['Reward']).to_csv(os.path.join(path, 'rewards_rotation.csv'), index=False)

    def save_model(self, path):
        torch.save(self.RotConv.state_dict(), path + '/_rot_conv_enc')
        torch.save(self.soft_q_net1.state_dict(), path + '/_q1_rot')
        torch.save(self.soft_q_net2.state_dict(), path + '/_q2_rot')
        torch.save(self.policy_net.state_dict(), path + '/_policy_rot')

    def save_best_model(self, path):
        torch.save(self.RotConv.state_dict(), path + '/_rot_best_conv_enc')
        torch.save(self.soft_q_net1.state_dict(), path + '/_q1_best_rot')
        torch.save(self.soft_q_net2.state_dict(), path + '/_q2_best_rot')
        torch.save(self.policy_net.state_dict(), path + '/_policy_best_rot')

    def load_model(self, path):
        self.RotConv.load_state_dict(torch.load(path + '/_rot_conv_enc'))
        self.soft_q_net1.load_state_dict(torch.load(path + '/_q1_rot'))
        self.soft_q_net2.load_state_dict(torch.load(path + '/_q2_rot'))
        self.policy_net.load_state_dict(torch.load(path + '/_policy_rot'))

        self.RotConv.eval()
        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()

    def load_model_for_grad_flow(self, path):
        self.RotConv.load_state_dict(torch.load(path + '/_rot_conv_enc'))
        self.soft_q_net1.load_state_dict(torch.load(path + '/_q1_rot'))
        self.soft_q_net2.load_state_dict(torch.load(path + '/_q2_rot'))
        self.policy_net.load_state_dict(torch.load(path + '/_policy_rot'))

    def load_best_model(self, path,device):
        if device == torch.device("cpu"):
            self.RotConv.load_state_dict(torch.load(path + '/_rot_best_conv_enc',map_location=torch.device('cpu')))
            self.soft_q_net1.load_state_dict(torch.load(path + '/_q1_best_rot',map_location=torch.device('cpu')))
            self.soft_q_net2.load_state_dict(torch.load(path + '/_q2_best_rot',map_location=torch.device('cpu')))
            self.policy_net.load_state_dict(torch.load(path + '/_policy_best_rot',map_location=torch.device('cpu')))
        else:
            self.RotConv.load_state_dict(torch.load(path + '/_rot_best_conv_enc'))
            self.soft_q_net1.load_state_dict(torch.load(path + '/_q1_best_rot'))
            self.soft_q_net2.load_state_dict(torch.load(path + '/_q2_best_rot'))
            self.policy_net.load_state_dict(torch.load(path + '/_policy_best_rot'))

        self.RotConv.eval()
        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()


class SAC_Trainer_rotation_w_grad_option_GCN_NO_LSTM():
    def __init__(self, replay_buffer, state_space, action_space, lr, grad, grad_value, hidden_dim, action_range):
        self.replay_buffer = replay_buffer
        self.lr = lr
        self.grad = grad
        self.grad_value = grad_value
        self.state_dimension=(state_space.shape[0])-1
        self.RotConv = RotConv(out_feat=(state_space.shape[0])-1).to(device)
        self.soft_q_net1 = QNetwork_NO_LSTM(state_space, action_space, hidden_dim).to(device)
        self.soft_q_net2 = QNetwork_NO_LSTM(state_space, action_space, hidden_dim).to(device)
        self.target_soft_q_net1 = QNetwork_NO_LSTM(state_space, action_space, hidden_dim).to(device)
        self.target_soft_q_net2 = QNetwork_NO_LSTM(state_space, action_space, hidden_dim).to(device)
        self.target_soft_q_net1.eval().requires_grad_(False)
        self.target_soft_q_net2.eval().requires_grad_(False)
        self.policy_net = SAC_PolicyNetwork_NO_LSTM_rotation(state_space, action_space, hidden_dim, action_range).to(device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)
        print('Soft Q Network (1,2): ', self.soft_q_net1)
        print('Policy Network: ', self.policy_net)

        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        soft_q_lr = self.lr
        policy_lr = self.lr
        alpha_lr = self.lr
        self.actor_loss_weight = policy_lr / soft_q_lr
        # Original lr
        # soft_q_lr = 3e-4
        # policy_lr = 3e-4
        # alpha_lr = 3e-4

        # soft_q_lr = 3e-5
        # policy_lr = 3e-5
        # alpha_lr = 3e-5

        # soft_q_lr = 3e-6
        # policy_lr = 3e-6
        # alpha_lr = 3e-6

        self.optimizer = optim.Adam(itertools.chain(self.RotConv.parameters(),
                                                    self.soft_q_net1.parameters(),
                                                    self.soft_q_net2.parameters(),
                                                    self.policy_net.parameters()),lr=soft_q_lr)

        # self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=soft_q_lr)
        # self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=soft_q_lr)
        # self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        # Original reward_scale = 10

    def update(self, path, batch_size, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99,
               soft_tau=1e-2):
        observation, action, reward, next_observation, done,residual_at_agent, next_residual_at_agent = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)
        observation = dgl.batch(observation)
        next_observation = dgl.batch(next_observation)
        residual_at_agent = th.FloatTensor(residual_at_agent).to(device)
        next_residual_at_agent = th.FloatTensor(next_residual_at_agent).to(device)
        state=(self.RotConv(observation)).reshape(batch_size,-1,self.state_dimension)
        state = (th.cat((state, residual_at_agent.unsqueeze(2)), dim=2)).to(device)
        with torch.no_grad():
            next_state = (self.RotConv(next_observation)).reshape(batch_size, -1, self.state_dimension)
            next_state = (th.cat((next_state, next_residual_at_agent.unsqueeze(2)), dim=2)).to(device)
        action = torch.FloatTensor(np.array(action)).to(device)

        reward = torch.FloatTensor(reward).unsqueeze(-1).to(
            device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done = torch.FloatTensor(np.float32(done)).unsqueeze(-1).to(device)
        # normalize with batch mean and std; plus a small number to prevent numerical problem
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(
            dim=0) + 1e-6)

        # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            # print('alpha loss: ',alpha_loss)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            with torch.no_grad():
                self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 1.
            alpha_loss = 0


        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2= self.soft_q_net2(state, action)
        with torch.no_grad():
            new_next_action, next_log_prob, _, _, _= self.policy_net.evaluate(next_state)
            # Training Q Function
            predict_target_q1= self.target_soft_q_net1(next_state, new_next_action )
            predict_target_q2 = self.target_soft_q_net2(next_state, new_next_action)
            target_q_min = torch.min(predict_target_q1, predict_target_q2) - self.alpha * next_log_prob
            target_q_value = reward + (1 - done) * gamma * target_q_min  # if done==1, only reward
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1.float(),
                                               target_q_value.float())  # detach: no gradients for the variable
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2.float(), target_q_value.float())
        q_loss = (q_value_loss1+q_value_loss2)/2.0

        # Training Policy Function
        predict_q1= self.soft_q_net1(state, new_action)
        predict_q2 = self.soft_q_net2(state, new_action)
        predicted_new_q_value = torch.min(predict_q1, predict_q2)

        predict_q1_critic_grad_only= self.soft_q_net1(state, new_action.detach())
        predict_q2_critic_grad_only= self.soft_q_net2(state, new_action.detach())

        predicted_new_q_value_critic_grad_only = torch.min(predict_q1_critic_grad_only, predict_q2_critic_grad_only)

        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()
        policy_loss_unbiased = policy_loss+ predicted_new_q_value_critic_grad_only.mean()
        loss = q_loss+self.actor_loss_weight*policy_loss_unbiased
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad == True:
            for param_group in self.optimizer.param_groups:
                nn.utils.clip_grad.clip_grad_norm_(param_group['params'],
                                                   max_norm=self.grad_value,
                                                   norm_type=2)

        self.optimizer.step()

        # print('q loss: ', q_value_loss1, q_value_loss2)
        # print('policy loss: ', policy_loss )

        # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        return q_loss.detach().cpu(), policy_loss.detach().cpu(), alpha_loss.detach().cpu(),loss.detach().cpu()

    def save_checkpoint(self, path, rewards):
        try:
            os.makedirs(path)
        except:
            pass

        torch.save(self.RotConv.state_dict(), path + '/_rot_conv_enc')
        torch.save(self.soft_q_net1.state_dict(), path + '/_q1_rot')
        torch.save(self.soft_q_net2.state_dict(), path + '/_q2_rot')
        torch.save(self.policy_net.state_dict(), path + '/_policy_rot')

        plt.cla()
        plt.plot(rewards, c='#bd0e3a', alpha=0.3)
        plt.plot(gaussian_filter1d(rewards, sigma=5), c='#bd0e3a', label='Rewards')
        plt.xlabel('Episodes')
        plt.ylabel('Cumulative reward')
        plt.title('Sac_LSTM_Rotation')
        plt.savefig(os.path.join(path, 'reward_rotation.png'))

        pd.DataFrame(rewards, columns=['Reward']).to_csv(os.path.join(path, 'rewards_rotation.csv'), index=False)

    def save_model(self, path):
        torch.save(self.RotConv.state_dict(), path + '/_rot_conv_enc')
        torch.save(self.soft_q_net1.state_dict(), path + '/_q1_rot')
        torch.save(self.soft_q_net2.state_dict(), path + '/_q2_rot')
        torch.save(self.policy_net.state_dict(), path + '/_policy_rot')

    def save_best_model(self, path):
        torch.save(self.RotConv.state_dict(), path + '/_rot_best_conv_enc')
        torch.save(self.soft_q_net1.state_dict(), path + '/_q1_best_rot')
        torch.save(self.soft_q_net2.state_dict(), path + '/_q2_best_rot')
        torch.save(self.policy_net.state_dict(), path + '/_policy_best_rot')

    def load_model(self, path):
        self.RotConv.load_state_dict(torch.load(path + '/_rot_conv_enc'))
        self.soft_q_net1.load_state_dict(torch.load(path + '/_q1_rot'))
        self.soft_q_net2.load_state_dict(torch.load(path + '/_q2_rot'))
        self.policy_net.load_state_dict(torch.load(path + '/_policy_rot'))

        self.RotConv.eval()
        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()

    def load_model_for_grad_flow(self, path):
        self.RotConv.load_state_dict(torch.load(path + '/_rot_conv_enc'))
        self.soft_q_net1.load_state_dict(torch.load(path + '/_q1_rot'))
        self.soft_q_net2.load_state_dict(torch.load(path + '/_q2_rot'))
        self.policy_net.load_state_dict(torch.load(path + '/_policy_rot'))

    def load_best_model(self, path):
        self.RotConv.load_state_dict(torch.load(path + '/_rot_best_conv_enc'))
        self.soft_q_net1.load_state_dict(torch.load(path + '/_q1_best_rot'))
        self.soft_q_net2.load_state_dict(torch.load(path + '/_q2_best_rot'))
        self.policy_net.load_state_dict(torch.load(path + '/_policy_best_rot'))

        self.RotConv.eval()
        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()

class SAC_Trainer_rotation_w_grad_option_GCN_trunc():
    def __init__(self, replay_buffer, state_space, action_space, lr, grad, grad_value, hidden_dim, action_range,trunc_seq_len):
        self.trunc_seq_len = trunc_seq_len
        self.replay_buffer = replay_buffer
        self.lr = lr
        self.grad = grad
        self.grad_value = grad_value
        self.state_dimension=(state_space.shape[0])-1
        self.RotConv = RotConv(out_feat=(state_space.shape[0])-1).to(device)
        self.soft_q_net1 = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.soft_q_net2 = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.target_soft_q_net1 = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.target_soft_q_net2 = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.target_soft_q_net1.eval().requires_grad_(False)
        self.target_soft_q_net2.eval().requires_grad_(False)
        self.policy_net = SAC_PolicyNetworkLSTM_rotation(state_space, action_space, hidden_dim, action_range).to(device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)
        print('Soft Q Network (1,2): ', self.soft_q_net1)
        print('Policy Network: ', self.policy_net)

        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        soft_q_lr = self.lr
        policy_lr = self.lr
        alpha_lr = self.lr
        self.actor_loss_weight = policy_lr / soft_q_lr
        # Original lr
        # soft_q_lr = 3e-4
        # policy_lr = 3e-4
        # alpha_lr = 3e-4

        # soft_q_lr = 3e-5
        # policy_lr = 3e-5
        # alpha_lr = 3e-5

        # soft_q_lr = 3e-6
        # policy_lr = 3e-6
        # alpha_lr = 3e-6

        self.optimizer = optim.Adam(itertools.chain(self.RotConv.parameters(),
                                                    self.soft_q_net1.parameters(),
                                                    self.soft_q_net2.parameters(),
                                                    self.policy_net.parameters()),lr=soft_q_lr)

        # self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=soft_q_lr)
        # self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=soft_q_lr)
        # self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        # Original reward_scale = 10

    def update(self, path, batch_size, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99,
               soft_tau=1e-2):
        hidden_in, hidden_out, observation, action, last_action, reward, next_observation, done,residual_at_agent, next_residual_at_agent = self.replay_buffer.sample(
            batch_size)
        # print('sample:', state, action,  reward, done)
        observation = dgl.batch(observation)
        observation= observation.to(device)
        next_observation = dgl.batch(next_observation)
        next_observation = next_observation.to(device)
        """residual at agent is:list(128) """
        residual_at_agent = th.FloatTensor(residual_at_agent).to(device)
        """ residual at agent after tensor tranform is: Tensor(128,100)"""
        next_residual_at_agent = th.FloatTensor(next_residual_at_agent).to(device)
        """ state after RotConv is a Tensor(128,100,20)"""
        state = self.RotConv(observation)
        shape_of_state = list(state.size())
        if ((shape_of_state[0]) != (batch_size*self.trunc_seq_len)):
            padded_state = th.zeros(batch_size*self.trunc_seq_len,shape_of_state[1]).to(device)
            padded_state[:shape_of_state[0],:] = state
            state = padded_state
        """ state after reshape is: Tensor(12800,20)"""
        state=(state).reshape(batch_size,-1,self.state_dimension)
        "state after concat is: Tensor(128,100,21)"
        state = (th.cat((state, residual_at_agent.unsqueeze(2)), dim=2)).to(device)
        with torch.no_grad():
            next_state = self.RotConv(next_observation)
            shape_of_next_state = list(next_state.size())
            if ((shape_of_next_state[0]) != (batch_size * self.trunc_seq_len)):
                padded_next_state = th.zeros(batch_size * self.trunc_seq_len, shape_of_next_state[1]).to(device)
                padded_next_state[:shape_of_next_state[0], :] = next_state
                next_state = padded_next_state

            next_state = (next_state).reshape(batch_size, -1, self.state_dimension)
            next_state = (th.cat((next_state, next_residual_at_agent.unsqueeze(2)), dim=2)).to(device)
        action = torch.FloatTensor(action).to(device)
        last_action = torch.FloatTensor(last_action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(-1).to(
            device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done = torch.FloatTensor(np.float32(done)).unsqueeze(-1).to(device)
        # normalize with batch mean and std; plus a small number to prevent numerical problem
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(
            dim=0) + 1e-6)

        # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q)
        new_action, log_prob, z, mean, log_std, _ = self.policy_net.evaluate(state, last_action, hidden_in)
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            # print('alpha loss: ',alpha_loss)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            with torch.no_grad():
                self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 1.
            alpha_loss = 0


        predicted_q_value1, _ = self.soft_q_net1(state, action, last_action, hidden_in)
        predicted_q_value2, _ = self.soft_q_net2(state, action, last_action, hidden_in)
        with torch.no_grad():
            new_next_action, next_log_prob, _, _, _, _ = self.policy_net.evaluate(next_state, action, hidden_out)
            # Training Q Function
            predict_target_q1, _ = self.target_soft_q_net1(next_state, new_next_action, action, hidden_out)
            predict_target_q2, _ = self.target_soft_q_net2(next_state, new_next_action, action, hidden_out)
            target_q_min = torch.min(predict_target_q1, predict_target_q2) - self.alpha * next_log_prob
            target_q_value = reward + (1 - done) * gamma * target_q_min  # if done==1, only reward
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1.float(),
                                               target_q_value.float())  # detach: no gradients for the variable
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2.float(), target_q_value.float())
        q_loss = (q_value_loss1+q_value_loss2)/2.0

        # Training Policy Function
        predict_q1, _ = self.soft_q_net1(state, new_action, last_action, hidden_in)
        predict_q2, _ = self.soft_q_net2(state, new_action, last_action, hidden_in)
        predicted_new_q_value = torch.min(predict_q1, predict_q2)

        predict_q1_critic_grad_only, _ = self.soft_q_net1(state, new_action.detach(), last_action.detach(), hidden_in)
        predict_q2_critic_grad_only, _ = self.soft_q_net2(state, new_action.detach(), last_action.detach(), hidden_in)
        predicted_new_q_value_critic_grad_only = torch.min(predict_q1_critic_grad_only, predict_q2_critic_grad_only)

        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()
        policy_loss_unbiased = policy_loss+ predicted_new_q_value_critic_grad_only.mean()
        loss = q_loss+self.actor_loss_weight*policy_loss_unbiased
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad == True:
            for param_group in self.optimizer.param_groups:
                nn.utils.clip_grad.clip_grad_norm_(param_group['params'],
                                                   max_norm=self.grad_value,
                                                   norm_type=2)

        self.optimizer.step()

        # print('q loss: ', q_value_loss1, q_value_loss2)
        # print('policy loss: ', policy_loss )

        # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        return q_loss.detach().cpu(), policy_loss.detach().cpu(), alpha_loss.detach().cpu(),loss.detach().cpu()

    def save_checkpoint(self, path, rewards):
        try:
            os.makedirs(path)
        except:
            pass

        torch.save(self.RotConv.state_dict(), path + '/_rot_conv_enc')
        torch.save(self.soft_q_net1.state_dict(), path + '/_q1_rot')
        torch.save(self.soft_q_net2.state_dict(), path + '/_q2_rot')
        torch.save(self.policy_net.state_dict(), path + '/_policy_rot')

        plt.cla()
        plt.plot(rewards, c='#bd0e3a', alpha=0.3)
        plt.plot(gaussian_filter1d(rewards, sigma=5), c='#bd0e3a', label='Rewards')
        plt.xlabel('Episodes')
        plt.ylabel('Cumulative reward')
        plt.title('Sac_LSTM_Rotation')
        plt.savefig(os.path.join(path, 'reward_rotation.png'))

        pd.DataFrame(rewards, columns=['Reward']).to_csv(os.path.join(path, 'rewards_rotation.csv'), index=False)

    def save_model(self, path):
        torch.save(self.RotConv.state_dict(), path + '/_rot_conv_enc')
        torch.save(self.soft_q_net1.state_dict(), path + '/_q1_rot')
        torch.save(self.soft_q_net2.state_dict(), path + '/_q2_rot')
        torch.save(self.policy_net.state_dict(), path + '/_policy_rot')

    def save_best_model(self, path):
        torch.save(self.RotConv.state_dict(), path + '/_rot_best_conv_enc')
        torch.save(self.soft_q_net1.state_dict(), path + '/_q1_best_rot')
        torch.save(self.soft_q_net2.state_dict(), path + '/_q2_best_rot')
        torch.save(self.policy_net.state_dict(), path + '/_policy_best_rot')

    def load_model(self, path):
        self.RotConv.load_state_dict(torch.load(path + '/_rot_conv_enc'))
        self.soft_q_net1.load_state_dict(torch.load(path + '/_q1_rot'))
        self.soft_q_net2.load_state_dict(torch.load(path + '/_q2_rot'))
        self.policy_net.load_state_dict(torch.load(path + '/_policy_rot'))

        self.RotConv.eval()
        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()

    def load_model_for_grad_flow(self, path):
        self.RotConv.load_state_dict(torch.load(path + '/_rot_conv_enc'))
        self.soft_q_net1.load_state_dict(torch.load(path + '/_q1_rot'))
        self.soft_q_net2.load_state_dict(torch.load(path + '/_q2_rot'))
        self.policy_net.load_state_dict(torch.load(path + '/_policy_rot'))

    def load_best_model(self, path):
        self.RotConv.load_state_dict(torch.load(path + '/_rot_best_conv_enc'))
        self.soft_q_net1.load_state_dict(torch.load(path + '/_q1_best_rot'))
        self.soft_q_net2.load_state_dict(torch.load(path + '/_q2_best_rot'))
        self.policy_net.load_state_dict(torch.load(path + '/_policy_best_rot'))

        self.RotConv.eval()
        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()


class SAC_Trainer_rotation_w_grad_option():
    def __init__(self, replay_buffer, state_space, action_space,lr,grad,grad_value, hidden_dim, action_range):
        self.replay_buffer = replay_buffer
        self.lr = lr
        self.grad=grad
        self.grad_value = grad_value
        self.soft_q_net1 = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.soft_q_net2 = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.target_soft_q_net1 = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.target_soft_q_net2 = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.policy_net = SAC_PolicyNetworkLSTM_rotation(state_space, action_space, hidden_dim, action_range).to(device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)
        print('Soft Q Network (1,2): ', self.soft_q_net1)
        print('Policy Network: ', self.policy_net)

        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        soft_q_lr = self.lr
        policy_lr = self.lr
        alpha_lr = self.lr

        # Original lr
        # soft_q_lr = 3e-4
        # policy_lr = 3e-4
        # alpha_lr = 3e-4

        # soft_q_lr = 3e-5
        # policy_lr = 3e-5
        # alpha_lr = 3e-5

        # soft_q_lr = 3e-6
        # policy_lr = 3e-6
        # alpha_lr = 3e-6

        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        # Original reward_scale = 10

    def update(self,path, batch_size, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99, soft_tau=1e-2):
        hidden_in, hidden_out, state, action, last_action, reward, next_state, done = self.replay_buffer.sample(
            batch_size)
        # print('sample:', state, action,  reward, done)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        last_action = torch.FloatTensor(last_action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(-1).to(
            device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done = torch.FloatTensor(np.float32(done)).unsqueeze(-1).to(device)

        predicted_q_value1, _ = self.soft_q_net1(state, action, last_action, hidden_in)
        predicted_q_value2, _ = self.soft_q_net2(state, action, last_action, hidden_in)
        new_action, log_prob, z, mean, log_std, _ = self.policy_net.evaluate(state, last_action, hidden_in)
        new_next_action, next_log_prob, _, _, _, _ = self.policy_net.evaluate(next_state, action, hidden_out)
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6)  # normalize with batch mean and std; plus a small number to prevent numerical problem
        # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q)
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            # print('alpha loss: ',alpha_loss)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 1.
            alpha_loss = 0

        # Training Q Function
        predict_target_q1, _ = self.target_soft_q_net1(next_state, new_next_action, action, hidden_out)
        predict_target_q2, _ = self.target_soft_q_net2(next_state, new_next_action, action, hidden_out)
        target_q_min = torch.min(predict_target_q1, predict_target_q2) - self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * gamma * target_q_min  # if done==1, only reward
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1.float(),target_q_value.float().detach())  # detach: no gradients for the variable
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2.float(), target_q_value.float().detach())

        if self.grad ==True:
            self.soft_q_optimizer1.zero_grad()
            q_value_loss1.backward()
            #plot_grad_flow(self.soft_q_net1.named_parameters(),"qvalue1",path)
            nn.utils.clip_grad_norm_(self.soft_q_net1.parameters(), max_norm=self.grad_value, norm_type=2)
            self.soft_q_optimizer1.step()
            self.soft_q_optimizer2.zero_grad()
            q_value_loss2.backward()
            #plot_grad_flow(self.soft_q_net2.named_parameters(),"qvalue2",path)
            nn.utils.clip_grad_norm_(self.soft_q_net2.parameters(), max_norm=self.grad_value, norm_type=2)
            self.soft_q_optimizer2.step()

            # Training Policy Function
            predict_q1, _ = self.soft_q_net1(state, new_action, last_action, hidden_in)
            predict_q2, _ = self.soft_q_net2(state, new_action, last_action, hidden_in)
            predicted_new_q_value = torch.min(predict_q1, predict_q2)
            policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=self.grad_value, norm_type=2)
            self.policy_optimizer.step()
        else:
            self.soft_q_optimizer1.zero_grad()
            q_value_loss1.backward()
            # plot_grad_flow(self.soft_q_net1.named_parameters(),"qvalue1",path)
            # nn.utils.clip_grad_norm_(self.soft_q_net1.parameters(), max_norm=50.0, norm_type=2)
            self.soft_q_optimizer1.step()
            self.soft_q_optimizer2.zero_grad()
            q_value_loss2.backward()
            # plot_grad_flow(self.soft_q_net2.named_parameters(),"qvalue2",path)
            # nn.utils.clip_grad_norm_(self.soft_q_net2.parameters(), max_norm=50.0, norm_type=2)
            self.soft_q_optimizer2.step()

            # Training Policy Function
            predict_q1, _ = self.soft_q_net1(state, new_action, last_action, hidden_in)
            predict_q2, _ = self.soft_q_net2(state, new_action, last_action, hidden_in)
            predicted_new_q_value = torch.min(predict_q1, predict_q2)
            policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()



        # print('q loss: ', q_value_loss1, q_value_loss2)
        # print('policy loss: ', policy_loss )

        # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        return q_value_loss1.detach().cpu(), q_value_loss2.detach().cpu(), policy_loss.detach().cpu(), alpha_loss.detach().cpu()



    def save_checkpoint(self, path, rewards):
        try:
            os.makedirs(path)
        except:
            pass

        torch.save(self.soft_q_net1.state_dict(), path + '/_q1_rot')
        torch.save(self.soft_q_net2.state_dict(), path + '/_q2_rot')
        torch.save(self.policy_net.state_dict(), path + '/_policy_rot')

        plt.cla()
        plt.plot(rewards, c='#bd0e3a', alpha=0.3)
        plt.plot(gaussian_filter1d(rewards, sigma=5), c='#bd0e3a', label='Rewards')
        plt.xlabel('Episodes')
        plt.ylabel('Cumulative reward')
        plt.title('Sac_LSTM_Rotation')
        plt.savefig(os.path.join(path, 'reward_rotation.png'))

        pd.DataFrame(rewards, columns=['Reward']).to_csv(os.path.join(path, 'rewards_rotation.csv'), index=False)

    def save_model(self, path):
        torch.save(self.soft_q_net1.state_dict(), path + '/_q1_rot')
        torch.save(self.soft_q_net2.state_dict(), path + '/_q2_rot')
        torch.save(self.policy_net.state_dict(), path + '/_policy_rot')

    def save_best_model(self, path):
        torch.save(self.soft_q_net1.state_dict(), path + '/_q1_best_rot')
        torch.save(self.soft_q_net2.state_dict(), path + '/_q2_best_rot')
        torch.save(self.policy_net.state_dict(), path + '/_policy_best_rot')

    def load_model(self, path):
        self.soft_q_net1.load_state_dict(torch.load(path + '/_q1_rot'))
        self.soft_q_net2.load_state_dict(torch.load(path + '/_q2_rot'))
        self.policy_net.load_state_dict(torch.load(path + '/_policy_rot'))

        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()

    def load_model_for_grad_flow(self, path):
        self.soft_q_net1.load_state_dict(torch.load(path + '/_q1_rot'))
        self.soft_q_net2.load_state_dict(torch.load(path + '/_q2_rot'))
        self.policy_net.load_state_dict(torch.load(path + '/_policy_rot'))



    def load_best_model(self, path):
        self.soft_q_net1.load_state_dict(torch.load(path + '/_q1_best_rot'))
        self.soft_q_net2.load_state_dict(torch.load(path + '/_q2_best_rot'))
        self.policy_net.load_state_dict(torch.load(path + '/_policy_best_rot'))

        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()

class SAC_Trainer_rotation():
    def __init__(self, replay_buffer, state_space, action_space, hidden_dim, action_range):
        self.replay_buffer = replay_buffer

        self.soft_q_net1 = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.soft_q_net2 = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.target_soft_q_net1 = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.target_soft_q_net2 = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.policy_net = SAC_PolicyNetworkLSTM_rotation(state_space, action_space, hidden_dim, action_range).to(device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)
        print('Soft Q Network (1,2): ', self.soft_q_net1)
        print('Policy Network: ', self.policy_net)

        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()
        # Original lr
        soft_q_lr = 3e-4
        policy_lr = 3e-4
        alpha_lr = 3e-4

        # soft_q_lr = 3e-5
        # policy_lr = 3e-5
        # alpha_lr = 3e-5

        # soft_q_lr = 3e-6
        # policy_lr = 3e-6
        # alpha_lr = 3e-6

        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        # Original reward_scale = 10

    def update(self,path, batch_size, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99, soft_tau=1e-2):
        hidden_in, hidden_out, state, action, last_action, reward, next_state, done = self.replay_buffer.sample(
            batch_size)
        # print('sample:', state, action,  reward, done)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        last_action = torch.FloatTensor(last_action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(-1).to(
            device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done = torch.FloatTensor(np.float32(done)).unsqueeze(-1).to(device)

        predicted_q_value1, _ = self.soft_q_net1(state, action, last_action, hidden_in)
        predicted_q_value2, _ = self.soft_q_net2(state, action, last_action, hidden_in)
        new_action, log_prob, z, mean, log_std, _ = self.policy_net.evaluate(state, last_action, hidden_in)
        new_next_action, next_log_prob, _, _, _, _ = self.policy_net.evaluate(next_state, action, hidden_out)
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6)  # normalize with batch mean and std; plus a small number to prevent numerical problem
        # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q)
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            # print('alpha loss: ',alpha_loss)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 1.
            alpha_loss = 0

        # Training Q Function
        predict_target_q1, _ = self.target_soft_q_net1(next_state, new_next_action, action, hidden_out)
        predict_target_q2, _ = self.target_soft_q_net2(next_state, new_next_action, action, hidden_out)
        target_q_min = torch.min(predict_target_q1, predict_target_q2) - self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * gamma * target_q_min  # if done==1, only reward
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1.float(),target_q_value.float().detach())  # detach: no gradients for the variable
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2.float(), target_q_value.float().detach())

        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        #plot_grad_flow(self.soft_q_net1.named_parameters(),"qvalue1",path)
        #nn.utils.clip_grad_norm_(self.soft_q_net1.parameters(), max_norm=50.0, norm_type=2)
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        #plot_grad_flow(self.soft_q_net2.named_parameters(),"qvalue2",path)
        #nn.utils.clip_grad_norm_(self.soft_q_net2.parameters(), max_norm=50.0, norm_type=2)
        self.soft_q_optimizer2.step()

        # Training Policy Function
        predict_q1, _ = self.soft_q_net1(state, new_action, last_action, hidden_in)
        predict_q2, _ = self.soft_q_net2(state, new_action, last_action, hidden_in)
        predicted_new_q_value = torch.min(predict_q1, predict_q2)
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # print('q loss: ', q_value_loss1, q_value_loss2)
        # print('policy loss: ', policy_loss )

        # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        return q_value_loss1.detach().cpu(), q_value_loss2.detach().cpu(), policy_loss.detach().cpu(), alpha_loss.detach().cpu()



    def save_checkpoint(self, path, rewards):
        try:
            os.makedirs(path)
        except:
            pass

        torch.save(self.soft_q_net1.state_dict(), path + '/_q1_rot')
        torch.save(self.soft_q_net2.state_dict(), path + '/_q2_rot')
        torch.save(self.policy_net.state_dict(), path + '/_policy_rot')

        plt.cla()
        plt.plot(rewards, c='#bd0e3a', alpha=0.3)
        plt.plot(gaussian_filter1d(rewards, sigma=5), c='#bd0e3a', label='Rewards')
        plt.xlabel('Episodes')
        plt.ylabel('Cumulative reward')
        plt.title('Sac_LSTM_Rotation')
        plt.savefig(os.path.join(path, 'reward_rotation.png'))

        pd.DataFrame(rewards, columns=['Reward']).to_csv(os.path.join(path, 'rewards_rotation.csv'), index=False)

    def save_model(self, path):
        torch.save(self.soft_q_net1.state_dict(), path + '/_q1_rot')
        torch.save(self.soft_q_net2.state_dict(), path + '/_q2_rot')
        torch.save(self.policy_net.state_dict(), path + '/_policy_rot')

    def save_best_model(self, path):
        torch.save(self.soft_q_net1.state_dict(), path + '/_q1_best_rot')
        torch.save(self.soft_q_net2.state_dict(), path + '/_q2_best_rot')
        torch.save(self.policy_net.state_dict(), path + '/_policy_best_rot')

    def load_model(self, path):
        self.soft_q_net1.load_state_dict(torch.load(path + '/_q1_rot'))
        self.soft_q_net2.load_state_dict(torch.load(path + '/_q2_rot'))
        self.policy_net.load_state_dict(torch.load(path + '/_policy_rot'))

        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()

    def load_model_for_grad_flow(self, path):
        self.soft_q_net1.load_state_dict(torch.load(path + '/_q1_rot'))
        self.soft_q_net2.load_state_dict(torch.load(path + '/_q2_rot'))
        self.policy_net.load_state_dict(torch.load(path + '/_policy_rot'))



    def load_best_model(self, path):
        self.soft_q_net1.load_state_dict(torch.load(path + '/_q1_best_rot'))
        self.soft_q_net2.load_state_dict(torch.load(path + '/_q2_best_rot'))
        self.policy_net.load_state_dict(torch.load(path + '/_policy_best_rot'))

        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()

class SAC_Trainer_translation():
    def __init__(self, replay_buffer, state_space, action_space, hidden_dim, action_range,path):
        self.replay_buffer = replay_buffer

        self.path3 = path + "/grad_flow_q1_images"
        try:
            os.makedirs(self.path3)
        except:
            pass

        self.path4 = path + "/grad_flow_q2_images"
        try:
            os.makedirs(self.path4)
        except:
            pass

        self.soft_q_net1 = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.soft_q_net2 = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.target_soft_q_net1 = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.target_soft_q_net2 = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        self.policy_net = SAC_PolicyNetworkLSTM(state_space, action_space, hidden_dim, action_range).to(device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)
        print('Soft Q Network (1,2): ', self.soft_q_net1)
        print('Policy Network: ', self.policy_net)

        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()
        # Original lr
        soft_q_lr = 3e-4
        policy_lr = 3e-4
        alpha_lr = 3e-4

        # soft_q_lr = 3e-5
        # policy_lr = 3e-5
        # alpha_lr = 3e-5

        # soft_q_lr = 3e-6
        # policy_lr = 3e-6
        # alpha_lr = 3e-6

        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        # Original reward_scale = 10

    def update(self,batch_size,eps,flag,clip_norm, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99, soft_tau=1e-2):
        hidden_in, hidden_out, state, action, last_action, reward, next_state, done = self.replay_buffer.sample(
            batch_size)
        # print('sample:', state, action,  reward, done)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        last_action = torch.FloatTensor(last_action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(-1).to(
            device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done = torch.FloatTensor(np.float32(done)).unsqueeze(-1).to(device)

        predicted_q_value1, _ = self.soft_q_net1(state, action, last_action, hidden_in)
        predicted_q_value2, _ = self.soft_q_net2(state, action, last_action, hidden_in)
        new_action, log_prob, z, mean, log_std, _ = self.policy_net.evaluate(state, last_action, hidden_in)
        new_next_action, next_log_prob, _, _, _, _ = self.policy_net.evaluate(next_state, action, hidden_out)
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6)  # normalize with batch mean and std; plus a small number to prevent numerical problem
        # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q)
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            # print('alpha loss: ',alpha_loss)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 1.
            alpha_loss = 0

        # Training Q Function
        predict_target_q1, _ = self.target_soft_q_net1(next_state, new_next_action, action, hidden_out)
        predict_target_q2, _ = self.target_soft_q_net2(next_state, new_next_action, action, hidden_out)
        target_q_min = torch.min(predict_target_q1, predict_target_q2) - self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * gamma * target_q_min  # if done==1, only reward
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1.float(),target_q_value.float().detach())  # detach: no gradients for the variable
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2.float(), target_q_value.float().detach())
        if flag ==True:
            self.soft_q_optimizer1.zero_grad()
            q_value_loss1.backward()
            # plot_grad_flow(self.soft_q_net1.named_parameters(),"qvalue1",path)
            self.plot_grad_flow(self.soft_q_net1.named_parameters(),"qvalue1",eps,False,clip_norm, self.path3)
            nn.utils.clip_grad_norm_(self.soft_q_net1.parameters(), max_norm=clip_norm, norm_type=2)
            self.plot_grad_flow(self.soft_q_net1.named_parameters(), "qvalue1",eps,True,clip_norm, self.path3)
            self.soft_q_optimizer1.step()
            self.soft_q_optimizer2.zero_grad()
            q_value_loss2.backward()

            # plot_grad_flow(self.soft_q_net2.named_parameters(),"qvalue2",path)
            self.plot_grad_flow(self.soft_q_net2.named_parameters(), "qvalue2",eps,False,clip_norm, self.path4)
            nn.utils.clip_grad_norm_(self.soft_q_net2.parameters(), max_norm=clip_norm, norm_type=2)
            self.plot_grad_flow(self.soft_q_net2.named_parameters(), "qvalue2",eps,True,clip_norm, self.path4)
            self.soft_q_optimizer2.step()

        else:
            self.soft_q_optimizer1.zero_grad()
            q_value_loss1.backward()
            #plot_grad_flow(self.soft_q_net1.named_parameters(),"qvalue1",path)
            nn.utils.clip_grad_norm_(self.soft_q_net1.parameters(), max_norm=clip_norm, norm_type=2)
            self.soft_q_optimizer1.step()
            self.soft_q_optimizer2.zero_grad()
            q_value_loss2.backward()
            #plot_grad_flow(self.soft_q_net2.named_parameters(),"qvalue2",path)
            nn.utils.clip_grad_norm_(self.soft_q_net2.parameters(), max_norm=clip_norm, norm_type=2)
            self.soft_q_optimizer2.step()

        # Training Policy Function
        predict_q1, _ = self.soft_q_net1(state, new_action, last_action, hidden_in)
        predict_q2, _ = self.soft_q_net2(state, new_action, last_action, hidden_in)
        predicted_new_q_value = torch.min(predict_q1, predict_q2)
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()


        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        #TODO Comment or uncomment for clippng policy network as well as Q network
        #nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=clip_norm, norm_type=2)
        self.policy_optimizer.step()

        # print('q loss: ', q_value_loss1, q_value_loss2)
        # print('policy loss: ', policy_loss )

        # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        return q_value_loss1.detach().cpu(), q_value_loss2.detach().cpu(), policy_loss.detach().cpu(), alpha_loss.detach().cpu()

    def plot_grad_flow(self,named_parameters, title_name, episode, clip, clip_norm, path):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in named_parameters:
            if (p.requires_grad) and ("bias" not in n):
                layers.append(n)
                # Load model for sac trainer sets the models to eval mode and therefore the .detach() attribture does not exist for networks in eval mode.
                ave_grads.append(p.grad.detach().cpu().abs().mean())
                max_grads.append(p.grad.detach().cpu().abs().max())
                # ave_grads.append(p.grad.cpu().abs().mean())
                # max_grads.append(p.grad.cpu().abs().mean())
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=-0.001, top=50)  # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title(title_name + "Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        clip_title = str(clip_norm)
        if clip == True:
            plt.savefig(os.path.join(path, title_name + '_epsiode' + str(
                episode) + '_after_clip' + clip_title + '_norm' + '.png'))
        else:
            plt.savefig(os.path.join(path, title_name + '_epsiode' + str(
                episode) + '_before_clip' + clip_title + '_norm' + '.png'))

    def save_checkpoint(self, path, rewards):
        try:
            os.makedirs(path)
        except:
            pass

        torch.save(self.soft_q_net1.state_dict(), path + '/_q1_trans')
        torch.save(self.soft_q_net2.state_dict(), path + '/_q2_trans')
        torch.save(self.policy_net.state_dict(), path + '/_policy_trans')

        plt.cla()
        plt.plot(rewards, c='#bd0e3a', alpha=0.3)
        plt.plot(gaussian_filter1d(rewards, sigma=5), c='#bd0e3a', label='Rewards')
        plt.xlabel('Episodes')
        plt.ylabel('Cumulative reward')
        plt.title('Sac_LSTM_Translation')
        plt.savefig(os.path.join(path, 'reward_translation.png'))

        pd.DataFrame(rewards, columns=['Reward']).to_csv(os.path.join(path, 'rewards_translation.csv'), index=False)

    def save_model(self, path):
        torch.save(self.soft_q_net1.state_dict(), path + '/_q1_trans')
        torch.save(self.soft_q_net2.state_dict(), path + '/_q2_trans')
        torch.save(self.policy_net.state_dict(), path + '/_policy_trans')

    def save_best_model(self, path):
        torch.save(self.soft_q_net1.state_dict(), path + '/_q1_best_trans')
        torch.save(self.soft_q_net2.state_dict(), path + '/_q2_best_trans')
        torch.save(self.policy_net.state_dict(), path + '/_policy_best_trans')

    def load_model(self, path):
        self.soft_q_net1.load_state_dict(torch.load(path + '/_q1_trans'))
        self.soft_q_net2.load_state_dict(torch.load(path + '/_q2_trans'))
        self.policy_net.load_state_dict(torch.load(path + '/_policy_trans'))

        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()

    def load_model_for_grad_flow(self, path):
        self.soft_q_net1.load_state_dict(torch.load(path + '/_q1_trans'))
        self.soft_q_net2.load_state_dict(torch.load(path + '/_q2_trans'))
        self.policy_net.load_state_dict(torch.load(path + '/_policy_trans'))



    def load_best_model(self, path):
        self.soft_q_net1.load_state_dict(torch.load(path + '/_q1_best_trans'))
        self.soft_q_net2.load_state_dict(torch.load(path + '/_q2_best_trans'))
        self.policy_net.load_state_dict(torch.load(path + '/_policy_best_trans'))

        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()
