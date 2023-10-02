import os
# import gymnasium as gym
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import random
import rl_utils
import collections
import itertools
from tqdm import tqdm
from scipy.stats import truncnorm
from torch.distributions import Normal
import imageio
import cv2
import sys 
sys.path.append(r'C:\Users\10696\Desktop\access\ma-gym')
from ma_gym.envs.combat.combat import Combat
import matplotlib.pyplot as plt

## 构造智能体 agent 的大脑，也就是输入状态，返回该状态下，选择每个动作的概率
## 输入是状态的，也就是 (车子center-point的坐标，车子的速度，杆的竖直角度，杆的角速度)
## 返回值应该是2 dim
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return F.softmax(self.fc3(x), dim=1)  ## 返回该状态下，选择的动作的概率

## 构造智能体 agent 的大脑，也就是输入状态，返回该状态下，每个动作的动作价值
## 输入是状态的，也就是 (车子center-point的坐标，车子的速度，杆的竖直角度，杆的角速度)
## 返回值应该是2 dim
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return self.fc3(x)


class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)   ##  策略网络的
        self.critic = ValueNet(state_dim, hidden_dim).to(device)       ##  价值网络
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)           ##  函数配置优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)         ##  价值函数配置优化器
        self.gamma = gamma          ## 衰减因子的呢
        self.lmbda = lmbda
        self.eps = eps  # PPO中截断范围的参数
        self.device = device
        self.epochs = epochs

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)       ## 拿到该状态下，每个动作的选择概率
        action_dist = torch.distributions.Categorical(probs)    ##   配置 好采样的概率
        action = action_dist.sample()        ## 对该状态下，所有的动作采样，采样的概率是probs
        return action.item()                 ## 返回依概率采样得到的动作

    def update(self, transition_dict):
        ## 拿到这条序列内的 状态、动作和奖励，下一个状态、是否完成的
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        ## 用下个状态求下一个状态的状态动作价值，然后间接求出当前状态的状态动作价值
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        ## 间接求出的价值 - 直接求出的当前状态的状态动作价值，也就是 TD-error，或者是优势函数 A
        td_delta = td_target - self.critic(states)
        ##  算出优势函数，广义优势估计，也就是每一步优势的均值
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        ## 选择的旧动作概率的log值，不反向传播求梯度，detach
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()
        for _ in range(self.epochs):  # 实现是包括了epoch数量的#
            ret = self.actor(states).gather(1, actions)
            log_probs = torch.log(ret)
            ratio = torch.exp(log_probs - old_log_probs)    ## 算重要性采样
            surr1 = ratio * advantage  ## 重要性采样和优势估计相乘的
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                    1 + self.eps) * advantage  # 截断
            ## 算出来的重要性采样，求出两者间的最小值，然后加负号，也就是最大化目标函数，不加负号的话是最小化目标函数
            # kk = torch.min(surr1, surr2)
            # kkk = -torch.sum(kk)
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            # kl_grad = torch.autograd.grad(actor_loss, surr1, create_graph=True, retain_graph=True)[0]
            # kl_grad2 = torch.autograd.grad(actor_loss, surr2, create_graph=True, retain_graph=True)[0]
            # kll = (kl_grad + kl_grad2 ) * advantage
            # kl_gra = torch.autograd.grad(actor_loss, ratio, create_graph=True, retain_graph=True)[0]
            # su = torch.sum(kl_gra!=kll)
            # kl_g = torch.autograd.grad(torch.sum(log_probs), ret, create_graph=True, retain_graph=True)[0]
            ## 直接求出当前状态的状态动作价值，和 间接求出的价值，使用 MSE 来算损失函数的，td_target不反向传播求梯度，detach
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()      ## 价值网络的参数梯度置零的
            actor_loss.backward()
            critic_loss.backward()                 ## 价值网络的损失loss反向传播梯度
            self.actor_optimizer.step() 
            self.critic_optimizer.step()           ## update 价值网络的参数

actor_lr = 3e-4
critic_lr = 1e-3
num_episodes = 100000
hidden_dim = 64
gamma = 0.99
lmbda = 0.97
eps = 0.2
epochs = 1
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

team_size = 2
grid_size = (15, 15)
#创建Combat环境，格子世界的大小为15x15，己方智能体和敌方智能体数量都为2
env = Combat(grid_shape=grid_size, n_agents=team_size, n_opponents=team_size)

state_dim = env.observation_space[0].shape[0]
action_dim = env.action_space[0].n
#两个智能体共享同一个策略
agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, 
            epochs, eps, gamma, device)

win_list = []
epoch = 10
allimage = []
for i in range(epoch):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            transition_dict_1 = {      ## 用来存放一个序列内的智能体1的数据
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': []
            }
            transition_dict_2 = {     ## 用来存放一个序列内的智能体2的数据
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': []
            }
            s = env.reset()
            terminal = False
            while not terminal:
                if 9900 < i_episode and i == epoch - 1:
                    img = env.render(mode = r'rgb_array')
                    allimage.append(img)
                a_1 = agent.take_action(s[0])                      ## 智能体1采取了动作的
                a_2 = agent.take_action(s[1])                      ## 智能体2采取了动作的
                next_s, r, done, info = env.step([a_1, a_2])       ## 环境执行动作的
                transition_dict_1['states'].append(s[0])           ## 加入智能体1的当前状态
                transition_dict_1['actions'].append(a_1)           ## 加入智能体1采取的动作
                transition_dict_1['next_states'].append(next_s[0]) ## 加入智能体1的下一个状态
                transition_dict_1['rewards'].append(               ## 加入智能体1的奖励，胜+100，负-0.1，胜的奖励值多了1000倍
                    r[0] + 100 if info['win'] else r[0] - 0.1)  
                transition_dict_1['dones'].append(False)           ## 没有完成的呢
                transition_dict_2['states'].append(s[1])           ## 加入智能体2的当前状态
                transition_dict_2['actions'].append(a_2)           ## 加入智能体2采取的动作
                transition_dict_2['next_states'].append(next_s[1]) ## 加入智能体2的下一个状态
                transition_dict_2['rewards'].append(               ## 加入智能体2的奖励，胜+100，负-0.1
                    r[1] + 100 if info['win'] else r[1] - 0.1)
                transition_dict_2['dones'].append(False)           ## 没有完成的呢
                s = next_s
                terminal = all(done)                               ## 是否都已经完成的
            win_list.append(1 if info["win"] else 0)
            ## 两个智能体默认 共享 同一个策略网络和价值网络
            agent.update(transition_dict_1)  ## 使用智能体1的数据来train策略网络和价值网络
            agent.update(transition_dict_2)  ## 使用智能体2的数据来train策略网络和价值网络
            if (i_episode + 1) % 100 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(win_list[-100:])
                })
            pbar.update(1)
# https://github.com/guicalare/Img2gif/blob/master/Code/Img2Gif.py
pth = r'C:\Users\10696\Desktop\access\Hands-on-RL'
imageio.mimsave(os.path.join(pth, 'chapter%s.gif'%str(20)), allimage, duration=100)

win_array = np.array(win_list)
#每100条轨迹取一次平均
win_array = np.mean(win_array.reshape(-1, 100), axis=1)

episodes_list = np.arange(win_array.shape[0]) * 100
plt.plot(episodes_list, win_array)
plt.xlabel('Episodes')
plt.ylabel('Win rate')
plt.title('IPPO on Combat')
plt.show()