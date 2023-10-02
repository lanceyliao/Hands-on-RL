import os
import sys
abspath = os.path.abspath(__file__)
filename = abspath.split(os.sep)[-1]
abspath = abspath.replace(filename, "")
sys.path.append(abspath)
filename = os.sep.join(abspath.split(os.sep)[-2:])
abspath = abspath.replace(filename, "")
sys.path.append(abspath)

import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils
from tqdm import tqdm
from net.fullconnect import fclayer
from net.activation import Softmax, ReLU
from net.loss import mean_square_two_derive_loss

## 构造智能体 agent 的大脑，也就是输入状态，返回该状态下，选择每个动作的概率
## 输入是状态的，也就是 (车子center-point的坐标，车子的速度，杆的竖直角度，杆的角速度)
## 返回值应该是2 dim
class PolicyNet():
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim, adam = True):
        super(PolicyNet, self).__init__()
        self.fc1 = fclayer(state_dim, hidden_dim, adam=adam)
        self.fc2 = fclayer(hidden_dim, action_dim, adam=adam)
        self.softmax = Softmax()
        self.relu = ReLU()

    def forward(self, x):
        self.x = x
        self.x1 = self.fc1.forward(x)
        self.x2 = self.relu.forward(self.x1)  # 隐藏层使用ReLU激活函数
        self.x3 = self.fc2.forward(self.x2)
        self.x6 = self.softmax.forward(self.x3, axis=-1)
        return self.x6
    
    def backward(self, delta):
        delta = self.softmax.backward(delta, self.x6)
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
    
## 构造智能体 agent 的大脑，也就是输入状态，返回该状态下，每个动作的动作价值
## 输入是状态的，也就是 (车子center-point的坐标，车子的速度，杆的竖直角度，杆的角速度)
## 返回值应该是2 dim
class ValueNet():
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, adam = True):
        super(ValueNet, self).__init__()
        self.fc1 = fclayer(state_dim, hidden_dim, adam=adam)
        self.fc2 = fclayer(hidden_dim, 1, adam=adam)
        self.relu = ReLU()
        self.param = {}

    def forward(self, x):
        self.x = x
        self.x1 = self.fc1.forward(x)
        self.x2 = self.relu.forward(self.x1)  # 隐藏层使用ReLU激活函数
        self.x3 = self.fc2.forward(self.x2)
        return self.x3

    def backward(self, delta):
        delta = self.fc2.backward(delta, inputs = self.x2)
        delta = self.relu.backward(delta)
        delta = self.fc1.backward(delta, inputs = self.x)
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

class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim) ##  策略网络的
        self.critic = ValueNet(state_dim, hidden_dim) ##  价值网络
        self.gamma = gamma   ## 衰减因子的呢
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

    def take_action(self, state):            # 根据动作概率分布随机采样
        state = np.array([state])
        probs = self.actor.forward(state)       ## 拿到该状态下，每个动作的选择概率
        action_dist = torch.distributions.Categorical(torch.tensor(probs))    ##   配置 好采样的概率
        action = action_dist.sample()        ## 对该状态下，所有的动作采样，采样的概率是probs
        return action.item()                 ## 返回依概率采样得到的动作
    
    def compute_advantage(self, gamma, lmbda, td_delta):
        td_delta = td_delta # .detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return np.array(advantage_list)
    
    def update(self, transition_dict):
        ## 拿到这条序列内的 奖励、状态和动作，下一个状态、是否完成的
        states = np.array(transition_dict['states'])
        actions = np.array(transition_dict['actions']).reshape((-1, 1))
        rewards = np.array(transition_dict['rewards']).reshape((-1, 1))
        next_states = np.array(transition_dict['next_states'])
        dones = np.array(transition_dict['dones']).reshape((-1, 1))
        ## 用下个状态求下一个状态的状态动作价值，然后间接求出当前状态的状态动作价值
        td_target = rewards + self.gamma * self.critic.forward(next_states) * (1 - dones)
        ## 间接求出的价值 - 直接求出的当前状态的状态动作价值，也就是 TD-error，或者是优势函数 A
        td_delta = td_target - self.critic.forward(states)
        ##  算出优势函数，广义优势估计，也就是每一步优势的均值
        advantage = self.compute_advantage(self.gamma, self.lmbda, td_delta)
        ## 选择的旧动作概率的log值，不反向传播求梯度，detach
        # old_log_probs = np.log(self.actor.forward(states).gather(1, actions))              # actions.detach()
        nowact = self.actor.forward(states)
        old_log_probs = []
        for ik in range(len(nowact)):
            old_log_probs.append(nowact[ik, actions[ik][0]])
        old_log_probs = np.expand_dims(np.array(old_log_probs), -1)
        old_log_probs = np.log(old_log_probs)

        for _ in range(self.epochs):
            ## 选择的动作概率的log值，不反向传播求梯度，detach
            # log_probs = torch.log(self.actor.forward(states).gather(1, actions))
            nowact = self.actor.forward(states)
            nowpro = []
            for ik in range(len(nowact)):
                nowpro.append(nowact[ik, actions[ik][0]])
            nowpro = np.expand_dims(np.array(nowpro), -1)
            log_probs = np.log(nowpro)
            ratio = np.exp(log_probs - old_log_probs) ## 算重要性采样
            surr1 = ratio * advantage  ## 重要性采样和优势估计相乘的
            surr2 = np.clip(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断
            ## 算出来的重要性采样，求出两者间的最小值，然后加负号，也就是最大化目标函数，不加负号的话是最小化目标函数
            actor_loss = np.mean(-np.min(np.concatenate([surr1, surr2], axis = 1), axis = 1, keepdims=True))  # PPO损失函数
            now_act_delta = np.zeros_like(nowact)
            num = surr1.size
            for i in range(len(surr1)):
                if surr1[i][0] < surr2[i][0]:
                    dt = (-1 / num) * advantage[i][0]
                elif surr1[i][0] == surr2[i][0]:
                    dt0 = -1 / num / 2.0  * advantage[i][0]
                    dt1 = -1 / num / 2.0  * advantage[i][0]
                    if ratio[i][0] > 1 + self.eps or ratio[i][0] < 1 - self.eps:
                        dt1 = 0
                    dt = dt0 + dt1
                else:
                    if ratio[i][0] > 1 + self.eps or ratio[i][0] < 1 - self.eps:
                        dt = 0
                    else:
                        dt = (-1 / num) * advantage[i][0]
                dt = dt * np.exp(log_probs[i][0] - old_log_probs[i][0]) * (1 / nowpro[i][0])
                now_act_delta[i, int(actions[i][0])] = dt
                # if actions[i][0] < 1:
                #     now_act_delta[i, 0] = dt
                # else:
                #     now_act_delta[i, 1] = dt
            
            self.actor.setzero()  ## 默认梯度会累积,这里需要显式将梯度置为0
            self.actor.backward(now_act_delta) ##  反向传播求出 actor 的梯度
            self.actor.update(lr=self.actor_lr)##  使用累加的梯度来update参数
            
            ## 直接求出当前状态的状态动作价值，和 间接求出的价值，使用 MSE 来算损失函数的，td_target不反向传播求梯度，detach
            nowval = self.critic.forward(states)
            # critic_loss = np.mean(F.mse_loss(nowval, td_target))       # td_target.detach()
            critic_loss, nowval_delta, _ = mean_square_two_derive_loss(nowval, td_target)    ## td_target.detach，不需要求梯度的
            self.critic.setzero()  ## 默认梯度会累积,这里需要显式将梯度置为0
            self.critic.backward(nowval_delta)  ##  反向传播求出 critic 的梯度，使用保存的参数6算梯度
            self.critic.update(self.critic_lr) ##  使用累加的梯度来update参数

actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 500
hidden_dim = 128
gamma = 0.98
lmbda = 0.95
epochs = 10
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = 'CartPole-v1'
env = gym.make(env_name, render_mode="rgb_array")
_ = env.reset(seed=0)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
            epochs, eps, gamma, device)

return_list = rl_utils.train_on_policy_agent_withpth(env, agent, num_episodes, 10, r'C:\Users\10696\Desktop\access\Hands-on-RL\numpy_RL_reinforcement_learning', '12CartPole')

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {}'.format(env_name))
plt.show()


## continuous 智能体
class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)  ## 用来求每个动作正态分布的均值
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim) ## 用来求每个动作正态分布的方差

    ## 要求出摆动作的概率分布，默认是正态分布的
    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = 2.0 * torch.tanh(self.fc_mu(x))       ## predict每个动作的正态分布的均值，取值区间则是 [-2, 2]
        std = F.softplus(self.fc_std(x))           ## predict每个动作的正态分布的方差，使用了函数 softplus
        return mu, std  # 高斯分布的均值和标准差


class PPOContinuous:
    ''' 处理连续动作的PPO算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        ## 状态的dimension，Pendulum-v1是 3 dim，也就是摆末端的(x,y)坐标，和角速度
        ## https://gymnasium.farama.org/environments/classic_control/pendulum/
        self.actor = PolicyNetContinuous(state_dim, hidden_dim,
                                         action_dim)  ##  策略网络
        self.critic = ValueNet(state_dim, hidden_dim)##  价值网络
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)  ## 策略网络配置优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)  ## 价值网络配置优化器
        self.gamma = gamma   ## 衰减因子的呢
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state): # 根据动作概率分布随机采样
        state = np.array([state])
        mu, std = self.actor(state)      ## 拿到该状态下，均值和标准差
        action_dist = torch.distributions.Normal(mu, std)    ##   配置 好采样的概率
        action = action_dist.sample()       ## 对该状态下，所有的动作采样，采样的分布是高斯分布
        return [action.item()]              ## 返回采样的动作值，也就是力矩的大小

    def update(self, transition_dict):
        ## 拿到这条序列内的 奖励、状态和动作，下一个状态、是否完成的
        states = np.array(transition_dict['states'],
                              dtype=torch.float)
        actions = np.array(transition_dict['actions'],
                               dtype=torch.float).reshape((-1, 1))
        rewards = np.array(transition_dict['rewards'],
                               dtype=torch.float).reshape((-1, 1))
        next_states = np.array(transition_dict['next_states'],
                                   dtype=torch.float)
        dones = np.array(transition_dict['dones'],
                             dtype=torch.float).reshape((-1, 1))
        rewards = (rewards + 8.0) / 8.0  # 和TRPO一样,对奖励进行修改,方便训练
        ## 用下个状态求下一个状态的状态动作价值，然后间接求出当前状态的状态动作价值
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        ## 间接求出的价值 - 直接求出的当前状态的状态动作价值，也就是 TD-error，或者是优势函数 A
        td_delta = td_target - self.critic(states)
        ##  算出优势函数值   广义优势估计，也就是平均值的
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta)
        mu, std = self.actor(states)   ## 拿到 continuous 动作分布的均值和方差
        ## 所有动作概率的分布，不反向传播求梯度，detach，给定正态分布的均值和方差，产生正态分布的
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        # 动作是正态分布
        old_log_probs = action_dists.log_prob(actions) ## 从产生的正态分布，给出每个动作的概率 log 值

        for _ in range(self.epochs):
            ## 拿到 continuous 动作分布的均值和方差
            mu, std = self.actor(states)
            ## 所有动作概率的分布，不反向传播求梯度，detach，给定正态分布的均值和方差，产生正态分布的
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(actions)  ## 从产生的正态分布，给出每个动作的概率 log 值
            ## 算重要性采样
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage   ## 重要性采样和优势估计相乘的
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断
            ## 算出来的重要性采样，求出两者间的最小值，然后加负号，也就是最大化目标函数，不加负号的话是最小化目标函数
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            ## 直接求出当前状态的状态动作价值，和 间接求出的价值，使用 MSE 来算损失函数的，td_target不反向传播求梯度，detach
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()   ## 价值网络的参数梯度置零的
            actor_loss.backward()
            critic_loss.backward()  ## 价值网络的损失loss反向传播梯度
            self.actor_optimizer.step()
            self.critic_optimizer.step()  # 更新价值函数

# actor_lr = 1e-4
# critic_lr = 5e-3
# num_episodes = 2000
# hidden_dim = 128
# gamma = 0.9
# lmbda = 0.9
# epochs = 10
# eps = 0.2
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# env_name = 'Pendulum-v1'
# env = gym.make(env_name, render_mode="rgb_array")
# _ = env.reset(seed=0)
# torch.manual_seed(0)
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.shape[0]  # 连续动作空间
# agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)

# return_list = rl_utils.train_on_policy_agent_withpth(env, agent, num_episodes, 10, r'C:\Users\10696\Desktop\access\Hands-on-RL\numpy_RL_reinforcement_learning', '12Pendulum')

# episodes_list = list(range(len(return_list)))
# plt.plot(episodes_list, return_list)
# plt.xlabel('Episodes')
# plt.ylabel('Returns')
# plt.title('PPO on {}'.format(env_name))
# plt.show()

# mv_return = rl_utils.moving_average(return_list, 21)
# plt.plot(episodes_list, mv_return)
# plt.xlabel('Episodes')
# plt.ylabel('Returns')
# plt.title('PPO on {}'.format(env_name))
# plt.show()