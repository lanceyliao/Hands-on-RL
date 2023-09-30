import os
import gymnasium as gym
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import rl_utils
import collections
import itertools
from tqdm import tqdm
from scipy.stats import truncnorm
from torch.distributions import Normal
import imageio
##  构造策略网络的
class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))  ## 激活函数的，>0，和relu类似
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  # rsample()是重参数化采样，单独的高斯分布采样不可导，重参数化可导的
        log_prob = dist.log_prob(normal_sample) ##  从分布内采样概率，并算出概率的log值
        action = torch.tanh(normal_sample)    ## 确定性策略输出的动作值，不是概率的，分布在 -1到1 之间
        # 计算tanh_normal分布的对数概率密度
        ## 熵的相反数
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        action = action * self.action_bound   ## 输出动作的，缩放到区间内
        return action, log_prob

## 状态动作价值网络，输出状态动作对 (状态，动作) 的价值
## 也就是动作价值网络
class QValueNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    ## 输入 (状态，动作)
    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1) # 拼接状态和动作
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)

class SACContinuous:
    ''' 处理连续动作的SAC算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound,
                 actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma,
                 device):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim,
                                         action_bound).to(device)  # 策略网络
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim,
                                            action_dim).to(device)  # 第一个Q网络
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim,
                                            action_dim).to(device)  # 第二个Q网络
        self.target_critic_1 = QValueNetContinuous(state_dim,
                                                   hidden_dim, action_dim).to(
                                                       device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNetContinuous(state_dim,
                                                   hidden_dim, action_dim).to(
                                                       device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        ## update 熵系数
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state)[0] ## 拿到确定性策略 网络的输出，也就是确定的动作
        return [action.item()]

    def calc_target(self, rewards, next_states, dones):  # 计算目标Q值
        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob ## 熵
        q1_value = self.target_critic_1(next_states, next_actions) ## 目标价值网络的下个状态的价值输出
        q2_value = self.target_critic_2(next_states, next_actions)
        ## 拿到最小的价值输出，然后加上熵，就是目标的下个状态的价值
        next_value = torch.min(q1_value,
                               q2_value) + self.log_alpha.exp() * entropy
        ## 目标当前状态的价值，时序差分的公式
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        ## 延迟update，EMA update，每次 update 很少的部分的
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        ## 拿到这条序列内的 状态、动作和奖励，下一个状态、是否完成的
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        # 和之前章节一样,对倒立摆环境的奖励进行重塑以便训练
        rewards = (rewards + 8.0) / 8.0

        # 更新两个Q网络
        ## 用下一个（状态、动作）对的动作价值 Q，然后间接求出当前的（状态、动作）对的动作价值
        td_target = self.calc_target(rewards, next_states, dones)
        ## 直接求出当前（状态、动作）的动作价值，和 间接求出的动作价值，使用 MSE 来算损失函数的，q_targets不反向传播求梯度
        critic_1_loss = torch.mean(
            F.mse_loss(self.critic_1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean(
            F.mse_loss(self.critic_2(states, actions), td_target.detach()))
        self.critic_1_optimizer.zero_grad() ## 价值网络的参数梯度置零的
        critic_1_loss.backward() ## 价值网络的损失loss反向传播梯度
        self.critic_1_optimizer.step() ## update 网络
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        ## self.actor(states) 是输出当前状态的动作，以及熵的相反数
        new_actions, log_prob = self.actor(states)
        entropy = -log_prob ## 熵
        ## 此时的动作价值网络已经 update 了的
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        ## update以后的动作价值网络，使用当前的（状态、动作）的动作价值，加了负号的，也就是最大化动作价值的
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

env_name = 'Pendulum-v1'
env = gym.make(env_name, render_mode="rgb_array")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]  # 动作最大值
random.seed(0)
np.random.seed(0)
_ = env.reset(seed=0)
torch.manual_seed(0)

actor_lr = 3e-4
critic_lr = 3e-3
alpha_lr = 3e-4
num_episodes = 100
hidden_dim = 128
gamma = 0.99
tau = 0.005  # 软更新参数
buffer_size = 100000
minimal_size = 1000
batch_size = 64
target_entropy = -env.action_space.shape[0]
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

import pickle
pth = r'C:\Users\10696\Desktop\access\Hands-on-RL\chapter18.pkl'

agent = SACContinuous(state_dim, hidden_dim, action_dim, action_bound,
                      actor_lr, critic_lr, alpha_lr, target_entropy, tau,
                      gamma, device)

if not os.path.exists(pth):
    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    ##  此处只会用来 train SAC智能体，并将智能体和环境交互产生的交互数据，加入到历史数据回放池
    return_list = rl_utils.train_off_policy_agent_withpth(env, agent, num_episodes,
                                                          replay_buffer, minimal_size,
                                                          batch_size, 10, r'C:\Users\10696\Desktop\access\Hands-on-RL', 18)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('SAC on {}'.format(env_name))
    plt.show()

    with open(pth, 'wb') as obj:
        pickle.dump([replay_buffer, episodes_list], obj)
else:
    with open(pth, 'rb') as obj:
        replay_buffer, episodes_list = pickle.load(obj)
    
## 网络结构和上面的SAC是相同的，主要区别在 train 的地方
class CQL:
    ''' CQL算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound,
                 actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma,
                 device, beta, num_random):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim,
                                         action_bound).to(device)  # 策略网络
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim,
                                            action_dim).to(device)  # 第一个Q网络
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim,
                                            action_dim).to(device)  # 第二个Q网络
        self.target_critic_1 = QValueNetContinuous(state_dim, hidden_dim,
                                                   action_dim).to(device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNetContinuous(state_dim, hidden_dim,
                                                   action_dim).to(device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        ## 配置网络的优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        ## update 熵系数
        self.log_alpha.requires_grad = True  #对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau

        self.beta = beta  # CQL损失函数中的系数
        self.num_random = num_random  # CQL中的动作采样数

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(device)
        action = self.actor(state)[0] ## 拿到确定性策略 网络的输出，也就是确定的动作
        return [action.item()]

    def soft_update(self, net, target_net):
        ## 延迟update，EMA update，每次 update 很少的部分的
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        ## 拿到这条序列内的 状态、动作和奖励，下一个状态、是否完成的
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).view(-1, 1).to(device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(device)
        rewards = (rewards + 8.0) / 8.0  # 对倒立摆环境的奖励进行重塑

        ## self.actor(states) 是输出下一个状态的动作，以及熵的相反数
        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob   ##  熵
        q1_value = self.target_critic_1(next_states, next_actions)        ##  目标价值网络的下个状态的价值输出
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value,   ## 拿到最小的价值输出，然后加上熵，就是目标的下个状态的价值
                               q2_value) + self.log_alpha.exp() * entropy
        ## 目标当前状态的价值，时序差分的公式
        td_target = rewards + self.gamma * next_value * (1 - dones)
        ## 直接求出当前（状态、动作）的动作价值，和 间接求出的动作价值，使用 MSE 来算损失函数的，q_targets不反向传播求梯度
        critic_1_loss = torch.mean(
            F.mse_loss(self.critic_1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean(
            F.mse_loss(self.critic_2(states, actions), td_target.detach()))


        # 以上与SAC相同,以下Q网络更新是CQL的额外部分
        '''
        考虑一般情况，希望Q在某个特定分布μ(s,a)上的期望值最小，也就是约束Q，最小化Q，减小Q的取值
        因特定分布并没有要求，所以什么样的分布都可以，下面的program指定了三类分布的
        1、动作是均匀分布的
        2、智能体策略网络给出采样的当前的状态的动作分布
        3、智能体策略网络给出采样的下一个状态的动作分布
        
        所以价值网络的输入 状态都是使用当前的，动作来自不同的分布（均匀分布的、策略网络给出当前状态的动作，策略网络给出下一个状态的动作）

        这三类不同的分布上，都要满足约束条件，也就是损失函数，达到减小价值Q的目的，而且
        
        q1_unif   价值1网络求出（当前的状态、均匀分布的动作）的动作价值
        q2_unif   价值2网络求出（当前的状态、均匀分布的动作）的动作价值
        q1_curr   价值1网络求出（当前的状态、策略网络给出的当前的动作）的动作价值
        q2_curr   价值2网络求出（当前的状态、策略网络给出的当前的动作）的动作价值
        q1_next   价值1网络求出（当前的状态、策略网络给出的下一个状态的动作）的动作价值
        q2_next   价值2网络求出（当前的状态、策略网络给出的下一个状态的动作）的动作价值

        实际任何动作分布都可以的，但是作者只使用了这三类(s,a)分布的，
        '''
        batch_size = states.shape[0]  ## 32*2
        ##  torch.rand 先随机生成数据，然后 uniform_均匀分布来填充tensor (-1, 1)，前一个 rand 其实只是初始化了tensor
        random_unif_actions = torch.rand(   ## (320, 1)
            [batch_size * self.num_random, actions.shape[-1]],
            dtype=torch.float).uniform_(-1, 1).to(device)
        random_unif_log_pi = np.log(0.5**next_actions.shape[-1])  ## 自定义的熵的相反数，和动作的个数相关，个数越少熵越小
        tmp_states = states.unsqueeze(1).repeat(1, self.num_random,     ## 多次重复当前的状态，最后reshape到2dim，和 states 的 dim 相同的  (320, 3)
                                                1).view(-1, states.shape[-1])
        tmp_next_states = next_states.unsqueeze(1).repeat(              ## 多次重复下一步状态，最后reshape到2dim，和 next_states 的 dim 相同的  (320, 3)
            1, self.num_random, 1).view(-1, next_states.shape[-1]) 
        random_curr_actions, random_curr_log_pi = self.actor(tmp_states)  ## 策略网络输出当前状态的动作，以及熵的相反数 (320, 1) (320, 1)
        random_next_actions, random_next_log_pi = self.actor(tmp_next_states) ## 策略网络输出下一个状态的动作，以及下一个熵的相反数 (320, 1) (320, 1)
        q1_unif = self.critic_1(tmp_states, random_unif_actions).view(    ## 价值1网络求出（当前的状态、均匀分布的动作）的动作价值   [64, 5, 1]
            -1, self.num_random, 1)
        q2_unif = self.critic_2(tmp_states, random_unif_actions).view(    ## 价值2网络求出（当前的状态、均匀分布的动作）的动作价值   [64, 5, 1] 
            -1, self.num_random, 1)
        q1_curr = self.critic_1(tmp_states, random_curr_actions).view(    ## 价值1网络求出（当前的状态、策略网络给出的当前的动作）的动作价值   [64, 5, 1]
            -1, self.num_random, 1)
        q2_curr = self.critic_2(tmp_states, random_curr_actions).view(    ## 价值2网络求出（当前的状态、策略网络给出的当前的动作）的动作价值   [64, 5, 1]
            -1, self.num_random, 1)
        q1_next = self.critic_1(tmp_states, random_next_actions).view(    ## 价值1网络求出（当前的状态、策略网络给出的下一个状态的动作）的动作价值   [64, 5, 1]
            -1, self.num_random, 1)
        q2_next = self.critic_2(tmp_states, random_next_actions).view(    ## 价值2网络求出（当前的状态、策略网络给出的下一个状态的动作）的动作价值   [64, 5, 1]
            -1, self.num_random, 1)
        q1_cat = torch.cat([              ## [64, 15, 1]
            ## 价值和熵相加，和动作的个数相关，个数越少熵越小
            q1_unif - random_unif_log_pi,
            q1_curr - random_curr_log_pi.detach().view(-1, self.num_random, 1),   ## 价值和熵相加
            q1_next - random_next_log_pi.detach().view(-1, self.num_random, 1)    ## 价值和熵相加
        ],
                           dim=1)
        q2_cat = torch.cat([                              ## [64, 15, 1]
            q2_unif - random_unif_log_pi,
            q2_curr - random_curr_log_pi.detach().view(-1, self.num_random, 1),   ## 价值和熵相加
            q2_next - random_next_log_pi.detach().view(-1, self.num_random, 1)    ## 价值和熵相加
        ],
                           dim=1)
        ## https://pytorch.org/docs/stable/generated/torch.logsumexp.html?highlight=logsumexp#torch.logsumexp
        ## 在用户指定的 dim 上，求exp，然后累加的，最后求log
        ## 迭代方程的第一个因子
        qf1_loss_1 = torch.logsumexp(q1_cat, dim=1).mean()
        qf2_loss_1 = torch.logsumexp(q2_cat, dim=1).mean()
        qf1_loss_2 = self.critic_1(states, actions).mean()  ## 返回历史真实数据的状态动作对的价值
        qf2_loss_2 = self.critic_2(states, actions).mean()  ## 返回历史真实数据的状态动作对的价值
        qf1_loss = critic_1_loss + self.beta * (qf1_loss_1 - qf1_loss_2) ## qf1_loss_2加了负号就是最大化，qf1_loss_1和critic_1_loss最小化
        qf2_loss = critic_2_loss + self.beta * (qf2_loss_1 - qf2_loss_2) ## qf2_loss_2加了负号就是最大化，qf2_loss_1和critic_2_loss最小化

        self.critic_1_optimizer.zero_grad()     ## 价值网络的参数梯度置零的
        qf1_loss.backward(retain_graph=True)    ## 价值网络的损失loss反向传播梯度
        self.critic_1_optimizer.step()          ## update 网络
        self.critic_2_optimizer.zero_grad()
        qf2_loss.backward(retain_graph=True)
        self.critic_2_optimizer.step()

        # 更新策略网络
        ## self.actor(states) 是输出当前状态的动作，以及熵的相反数
        new_actions, log_prob = self.actor(states)
        entropy = -log_prob  ## 熵
        ## 此时的动作价值网络已经 update 了的
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        ## update以后的动作价值网络，使用当前的（状态、动作）的动作价值，加了负号的，也就是最大化动作价值的
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

random.seed(0)               ## 配置随机种子方便复现的
np.random.seed(0)
_ = env.reset(seed=0)
torch.manual_seed(0)

beta = 5.0
num_random = 5
num_epochs = 100
num_trains_per_epoch = 500

agent = CQL(state_dim, hidden_dim, action_dim, action_bound, actor_lr,         ## 实例化 CQL
            critic_lr, alpha_lr, target_entropy, tau, gamma, device, beta,
            num_random)

return_list = []
allimage = []
for i in range(10):
    with tqdm(total=int(num_epochs / 10), desc='Iteration %d' % i) as pbar:
        for i_epoch in range(int(num_epochs / 10)):
            # 此处与环境交互只是为了评估策略,最后作图用,不会用于训练
            epoch_return = 0
            state = env.reset()
            if len(state)!=2*2-1:
                state = state[0]
            done = False
            while not done:
                if i==9:
                    img = env.render()
                    allimage.append(img)
                action = agent.take_action(state)
                ##  环境执行动作，并反馈下一个状态、动作的奖励、是否完成、步长太长的，info
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated | truncated       ## 终止或者步长太长，都会导致已经结束
                # next_state, reward, done, _ = env.step(action)
                state = next_state
                epoch_return += reward
            return_list.append(epoch_return)

            for _ in range(num_trains_per_epoch):
                ## 从SAC在train时保存的历史数据采样，用来 train CQL
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                transition_dict = {
                    'states': b_s,
                    'actions': b_a,
                    'next_states': b_ns,
                    'rewards': b_r,
                    'dones': b_d
                }
                agent.update(transition_dict)  ## train CQL

            if (i_epoch + 1) % 10 == 0:
                pbar.set_postfix({
                    'epoch':
                    '%d' % (num_epochs / 10 * i + i_epoch + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)
    if i==9:
        # https://github.com/guicalare/Img2gif/blob/master/Code/Img2Gif.py
        pth = r'C:\Users\10696\Desktop\access\Hands-on-RL'
        imageio.mimsave(os.path.join(pth, 'chapter%s.gif'%str(18)), allimage, duration=10)

epochs_list = list(range(len(return_list)))
plt.plot(epochs_list, return_list)
plt.xlabel('Epochs')
plt.ylabel('Returns')
plt.title('CQL on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('CQL on {}'.format(env_name))
plt.show()