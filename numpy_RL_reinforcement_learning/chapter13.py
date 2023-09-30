import os
import sys
abspath = os.path.abspath(__file__)
filename = abspath.split(os.sep)[-1]
abspath = abspath.replace(filename, "")
sys.path.append(abspath)
filename = os.sep.join(abspath.split(os.sep)[-2:])
abspath = abspath.replace(filename, "")
sys.path.append(abspath)

import random
import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils
from tqdm import tqdm
from net.fullconnect import fclayer
from net.activation import Softmax, ReLU, tanh
from net.loss import mean_square_two_derive_loss

##  构造策略网络的
class PolicyNet():
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim, adam = True):
        super(PolicyNet, self).__init__()
        self.fc1 = fclayer(state_dim, hidden_dim, adam=adam)
        self.fc2 = fclayer(hidden_dim, action_dim, adam=adam)
        self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值
        self.tanh = tanh()
        # self.tanh = torch.nn.Tanh().requires_grad_(True)
        self.relu = ReLU()

    def forward(self, x):
        self.x = x
        self.x1 = self.fc1.forward(x)
        self.x2 = self.relu.forward(self.x1)  # 隐藏层使用ReLU激活函数
        self.x3 = self.fc2.forward(self.x2)
        self.x6 = self.tanh.forward(self.x3)
        # self.k = torch.tensor(self.x3).requires_grad_(True)
        # self.x6 = self.tanh(self.k).requires_grad_(True)
        # self.x00 = self.x6.detach().cpu().numpy() * self.action_bound
        self.x00 = self.x6 * self.action_bound
        return self.x00
    
    def backward(self, delta):
        delta = delta * self.action_bound
        delta = self.tanh.backward(delta)
        # kk = torch.sum(self.x6).requires_grad_(True)
        # dd = torch.autograd.grad(kk, self.k, create_graph=True)[0].detach().cpu().numpy()
        # delta = delta * dd
        delta = self.fc2.backward(delta, inputs=self.x2)
        delta = self.relu.backward(delta)
        delta = self.fc1.backward(delta, inputs=self.x)
        return delta
    
    def update(self, lr):
        self.fc2.update(lr)
        self.fc1.update(lr)
    
    def setzero(self):
        self.fc2.setzero()
        self.fc1.setzero()

    def save_model(self):
        return [self.fc1.save_model(), self.fc2.save_model()]
        
    def restore_model(self, models):
        self.fc1.restore_model(models[0])
        self.fc2.restore_model(models[1])
    
    def setparam(self, model):
        self.fc1.params = model.fc1.params.copy()
        self.fc2.params = model.fc2.params.copy()
        if self.fc1.bias:
            self.fc1.bias_params = model.fc1.bias_params.copy()
        if self.fc2.bias:
            self.fc2.bias_params = model.fc2.bias_params.copy()

## 状态动作价值网络，输出状态动作对 (状态，动作) 的价值
## 也就是动作价值网络
class QValueNet():
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim, adam = True):
        super(QValueNet, self).__init__()
        self.fc1 = fclayer(state_dim + action_dim, hidden_dim, adam=adam)
        self.fc2 = fclayer(hidden_dim, hidden_dim, adam=adam)
        self.fc_out = fclayer(hidden_dim, 1, adam=adam)
        self.relu0 = ReLU()
        self.relu1 = ReLU()

    ## 输入 (状态，动作)
    def forward(self, x, a):
        cat = np.concatenate([x, a], axis=1) # 拼接状态和动作
        self.x = cat
        self.x1 = self.fc1.forward(self.x)
        self.x2 = self.relu0.forward(self.x1)  # 隐藏层使用ReLU激活函数
        self.x3 = self.fc2.forward(self.x2)
        self.x6 = self.relu1.forward(self.x3)
        self.x00 = self.fc_out.forward(self.x6)
        return self.x00
    
    def backward(self, delta):
        delta = self.fc_out.backward(delta, inputs=self.x6)
        delta = self.relu1.backward(delta)
        delta = self.fc2.backward(delta, inputs=self.x2)
        delta = self.relu0.backward(delta)
        delta = self.fc1.backward(delta, inputs=self.x)
        return delta
    
    def update(self, lr):
        self.fc_out.update(lr)
        self.fc2.update(lr)
        self.fc1.update(lr)
    
    def setzero(self):
        self.fc_out.setzero()
        self.fc2.setzero()
        self.fc1.setzero()

    def save_model(self):
        return [self.fc1.save_model(), self.fc2.save_model(), self.fc_out.save_model()]
        
    def restore_model(self, models):
        self.fc1.restore_model(models[0])
        self.fc2.restore_model(models[1])
        self.fc_out.restore_model(models[2])
    
    def setparam(self, model):
        self.fc1.params = model.fc1.params.copy()
        self.fc2.params = model.fc2.params.copy()
        self.fc_out.params = model.fc_out.params.copy()
        if self.fc1.bias:
            self.fc1.bias_params = model.fc1.bias_params.copy()
        if self.fc2.bias:
            self.fc2.bias_params = model.fc2.bias_params.copy()
        if self.fc_out.bias:
            self.fc_out.bias_params = model.fc_out.bias_params.copy()

class DDPG:
    ''' DDPG算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound) ##  策略网络的
        self.critic = QValueNet(state_dim, hidden_dim, action_dim) ## 状态动作价值网络
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound) ##  目标策略网络，延迟update
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim) ##  目标状态动作价值网络，延迟update
        # 初始化目标价值网络并设置和价值网络相同的参数
        self.target_critic.setparam(self.critic)
        # 初始化目标策略网络并设置和策略相同的参数
        self.target_actor.setparam(self.actor)
        self.gamma = gamma  ## 折扣因子
        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.tau = tau  # 目标网络软更新参数
        self.action_dim = action_dim  ## 动作的dim
        self.actor_lr  = actor_lr
        self.critic_lr = critic_lr

    def take_action(self, state):
        state = np.array([state])
        action = self.actor.forward(state)   ## 拿到确定性策略 网络的输出，也就是确定的动作
        # 给动作添加噪声，增加探索
        action = action + self.sigma * np.random.randn(self.action_dim)
        return action

    def soft_update_Value(self, net, target_net):   ## EMA的方式update目标网络的，每次update很少的内容
        target_net.fc1.params = (target_net.fc1.params * (1.0 - self.tau) + net.fc1.params * self.tau)
        target_net.fc2.params = (target_net.fc2.params * (1.0 - self.tau) + net.fc2.params * self.tau)
        target_net.fc_out.params = (target_net.fc_out.params * (1.0 - self.tau) + net.fc_out.params * self.tau)
        
        target_net.fc1.bias_params = (target_net.fc1.bias_params * (1.0 - self.tau) + net.fc1.bias_params * self.tau)
        target_net.fc2.bias_params = (target_net.fc2.bias_params * (1.0 - self.tau) + net.fc2.bias_params * self.tau)
        target_net.fc_out.bias_params = (target_net.fc_out.bias_params * (1.0 - self.tau) + net.fc_out.bias_params * self.tau)

    def soft_update_Policy(self, net, target_net):   ## EMA的方式update目标网络的，每次update很少的内容
        target_net.fc1.params = (target_net.fc1.params * (1.0 - self.tau) + net.fc1.params * self.tau)
        target_net.fc2.params = (target_net.fc2.params * (1.0 - self.tau) + net.fc2.params * self.tau)
        
        target_net.fc1.bias_params = (target_net.fc1.bias_params * (1.0 - self.tau) + net.fc1.bias_params * self.tau)
        target_net.fc2.bias_params = (target_net.fc2.bias_params * (1.0 - self.tau) + net.fc2.bias_params * self.tau)

    def update(self, transition_dict):
        ## 拿到这条序列内的 状态、动作和奖励，下一个状态、是否完成的
        states = np.array(transition_dict['states'])
        actions = np.array(transition_dict['actions']).reshape((-1, 1))
        rewards = np.array(transition_dict['rewards']).reshape((-1, 1))
        next_states = np.array(transition_dict['next_states'])
        dones = np.array(transition_dict['dones']).reshape((-1, 1))

        ## 下个状态+下个状态的动作，目标状态动作价值网络 输出（状态，动作）对的动作价值 Q 
        next_q_values = self.target_critic.forward(next_states, self.target_actor.forward(next_states))
        ## 用下一个（状态、动作）对的动作价值 Q，然后间接求出当前的（状态、动作）对的动作价值
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        ## 直接求出当前（状态、动作）的动作价值，和 间接求出的动作价值，使用 MSE 来算损失函数的，q_targets不反向传播求梯度
        nowval = self.critic.forward(states, actions)
        critic_loss, nowval_delta, _ = mean_square_two_derive_loss(nowval, q_targets)    ## td_target.detach，不需要求梯度的
        self.critic.setzero()  ## 默认梯度会累积,这里需要显式将梯度置为0
        self.critic.backward(nowval_delta)  ##  反向传播求出 critic 的梯度
        self.critic.update(self.critic_lr) ##  使用累加的梯度来update参数

        ## update以后的动作价值网络，使用当前的（状态、动作）的动作价值，加了负号的，也就是最大化动作价值的
        ## self.actor(states) 是输出当前状态的动作
        '''
         确定性策略网络的 update，稍稍复杂些，嵌套了两个网络，首先用当前的状态和策略网络，输出当前状态的动作predict_A=self.actor(states)，
         然后 self.critic(states, self.actor(states))输出了当前（状态、动作）的动作价值Q，并求均值加负号，也就是最大化动作价值，
         反向传播用来update 策略网络actor，因critic动作价值网络上面已经update过了，所以下面就只update策略网络actor，critic网络不变的。
        '''
        actret = self.critic.forward(states, self.actor.forward(states))
        num = 1
        for i in actret.shape:
            num = num * i
        actor_loss = -np.mean(actret)
        delta = np.ones_like(actret) * (-1) / num
        self.critic.setzero()
        delta = self.critic.backward(delta)
        delta = np.expand_dims(delta[:, -1], -1)

        self.actor.setzero()  ## 默认梯度会累积,这里需要显式将梯度置为0
        self.actor.backward(delta)  ##  反向传播求出 critic 的梯度
        self.actor.update(self.actor_lr) ##  使用累加的梯度来update参数
        ## 延迟少量的update网络，也就是EMA的方式
        self.soft_update_Policy(self.actor, self.target_actor)  # 软更新策略网络
        self.soft_update_Value(self.critic, self.target_critic)  # 软更新价值网络

actor_lr = 3e-4
critic_lr = 3e-3
num_episodes = 200
hidden_dim = 64
gamma = 0.98
tau = 0.005  # 软更新参数
buffer_size = 10000
minimal_size = 1000
batch_size = 64
sigma = 0.01  # 高斯噪声标准差

env_name = 'Pendulum-v1'
env = gym.make(env_name, render_mode="rgb_array")
random.seed(0)
np.random.seed(0)
# _ = env.reset(seed=0)
replay_buffer = rl_utils.ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]  # 动作最大值
agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma)

return_list = rl_utils.train_off_policy_agent_withpth(env, agent, num_episodes, replay_buffer, minimal_size, batch_size, \
    6, r'C:\Users\10696\Desktop\access\Hands-on-RL\numpy_RL_reinforcement_learning', '13')

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DDPG on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DDPG on {}'.format(env_name))
plt.show()