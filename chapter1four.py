import random
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import rl_utils
from tqdm import tqdm
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

# env_name = 'Pendulum-v1'
# env = gym.make(env_name, render_mode="rgb_array")
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.shape[0]
# action_bound = env.action_space.high[0]  # 动作最大值
# random.seed(0)
# np.random.seed(0)
# _ = env.reset(seed=0)
# torch.manual_seed(0)

# actor_lr = 3e-4
# critic_lr = 3e-3
# alpha_lr = 3e-4
# num_episodes = 100
# hidden_dim = 128
# gamma = 0.99
# tau = 0.005  # 软更新参数
# buffer_size = 100000
# minimal_size = 1000
# batch_size = 64
# target_entropy = -env.action_space.shape[0]
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
#     "cpu")

# replay_buffer = rl_utils.ReplayBuffer(buffer_size)
# agent = SACContinuous(state_dim, hidden_dim, action_dim, action_bound,
#                       actor_lr, critic_lr, alpha_lr, target_entropy, tau,
#                       gamma, device)

# return_list = rl_utils.train_off_policy_agent_withpth(env, agent, num_episodes,
#                                               replay_buffer, minimal_size,
#                                               batch_size, 10, r'C:\Users\10696\Desktop\access\Hands-on-RL', '1four')

# episodes_list = list(range(len(return_list)))
# plt.plot(episodes_list, return_list)
# plt.xlabel('Episodes')
# plt.ylabel('Returns')
# plt.title('SAC on {}'.format(env_name))
# plt.show()

# mv_return = rl_utils.moving_average(return_list, 9)
# plt.plot(episodes_list, mv_return)
# plt.xlabel('Episodes')
# plt.ylabel('Returns')
# plt.title('SAC on {}'.format(env_name))
# plt.show()

## 构造智能体 agent 的大脑，也就是输入状态，返回该状态下，选择每个动作的概率
## 输入是状态的，也就是 (车子center-point的坐标，车子的速度，杆的竖直角度，杆的角速度)
## 返回值应该是2 dim
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)  ## 返回该状态下，选择的动作的概率，离散

## 构造智能体 agent 的大脑，也就是输入状态，返回该状态下，每个动作的动作价值
## 输入是状态的，也就是 (车子center-point的坐标，车子的速度，杆的竖直角度，杆的角速度)
## 返回值应该是2 dim
class QValueNet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class SAC:
    ''' 处理离散动作的SAC算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 alpha_lr, target_entropy, tau, gamma, device):
        # 策略网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        # 第一个Q网络
        self.critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        # 第二个Q网络
        self.critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1 = QValueNet(state_dim, hidden_dim,
                                         action_dim).to(device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNet(state_dim, hidden_dim,
                                         action_dim).to(device)  # 第二个目标Q网络
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
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def take_action(self, state):            # 根据动作概率分布随机采样
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)       ## 拿到该状态下，每个动作的选择概率
        action_dist = torch.distributions.Categorical(probs)    ##   配置 好采样的概率
        action = action_dist.sample()        ## 对该状态下，所有的动作采样，采样的概率是probs
        return action.item()                 ## 返回依概率采样得到的动作
        
    # 计算目标Q值,直接用策略网络的输出概率进行期望计算
    def calc_target(self, rewards, next_states, dones):
        next_probs = self.actor(next_states) ## 动作的概率
        next_log_probs = torch.log(next_probs + 1e-8)  ## 熵的相反数
        ## 动作概率密度 乘 熵，就是熵的期望
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)  ## 熵的期望的呢
        q1_value = self.target_critic_1(next_states) ## 目标价值网络的下个状态的价值输出
        q2_value = self.target_critic_2(next_states)
        ## 拿到最小的状态价值，并乘动作概率，也就是动作价值
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)
        ## 然后加上熵
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        ## 就是目标的下个状态的价值
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        ## 延迟update，EMA update，每次 update 很少的部分的
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)  # 动作不再是float类型
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # 更新两个Q网络
        ## 用下一个（状态、动作）对的动作价值 Q，然后间接求出当前的（状态、动作）对的动作价值
        td_target = self.calc_target(rewards, next_states, dones)
        ## 直接求出当前（状态、动作）的动作价值，和 间接求出的动作价值，使用 MSE 来算损失函数的，q_targets不反向传播求梯度
        critic_1_q_values = self.critic_1(states).gather(1, actions)
        critic_1_loss = torch.mean(
            F.mse_loss(critic_1_q_values, td_target.detach()))
        critic_2_q_values = self.critic_2(states).gather(1, actions)
        critic_2_loss = torch.mean(
            F.mse_loss(critic_2_q_values, td_target.detach()))
        self.critic_1_optimizer.zero_grad() ## 价值网络的参数梯度置零的
        critic_1_loss.backward() ## 价值网络的损失loss反向传播梯度
        self.critic_1_optimizer.step() ## update 网络
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        ## self.actor(states) 是输出当前状态的动作，以及熵的相反数
        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8) ## 熵的相反数
        # 直接根据概率计算熵
        ## 算熵的期望也就是：动作概率 * 熵的相反数
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        ## 此时的动作价值网络已经 update 了的
        q1_value = self.critic_1(states)
        q2_value = self.critic_2(states)
        ## 拿到最小的状态价值，并乘动作概率，也就是动作价值
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)  # 直接根据概率计算期望
        ## update以后的动作价值网络，使用当前的（状态、动作）的动作价值，加了负号的，也就是最大化动作价值的
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)


actor_lr = 1e-3
critic_lr = 1e-2
alpha_lr = 1e-2
num_episodes = 200
hidden_dim = 128
gamma = 0.98
tau = 0.005  # 软更新参数
buffer_size = 10000
minimal_size = 500
batch_size = 64
target_entropy = -1
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

env_name = 'CartPole-v1'
env = gym.make(env_name, render_mode="rgb_array")
random.seed(0)
np.random.seed(0)
_ = env.reset(seed=0)
torch.manual_seed(0)
replay_buffer = rl_utils.ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = SAC(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, alpha_lr,
            target_entropy, tau, gamma, device)

return_list = rl_utils.train_off_policy_agent_withpth(env, agent, num_episodes,
                                              replay_buffer, minimal_size,
                                              batch_size, 10, r'C:\Users\10696\Desktop\access\Hands-on-RL', '1fourCartPole')

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('SAC on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('SAC on {}'.format(env_name))
plt.show()