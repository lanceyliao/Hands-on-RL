import random
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils
from tqdm import tqdm
import imageio

##  构造策略网络的
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.action_bound  ## 直接返回动作的大小

## 状态动作价值网络，输出状态动作对 (状态，动作) 的价值
## 也就是动作价值网络
class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    ## 输入 (状态，动作)
    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1) # 拼接状态和动作
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)

class DDPG:
    ''' DDPG算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device) ##  策略网络的
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device) ## 状态动作价值网络
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device) ##  目标策略网络，延迟update
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim).to(device) ##  目标状态动作价值网络，延迟update
        # 初始化目标价值网络并设置和价值网络相同的参数
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化目标策略网络并设置和策略相同的参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma  ## 折扣因子
        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.tau = tau  # 目标网络软更新参数
        self.action_dim = action_dim  ## 动作的dim
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state).item()   ## 拿到确定性策略 网络的输出，也就是确定的动作
        # 给动作添加噪声，增加探索
        action = action + self.sigma * np.random.randn(self.action_dim)
        return action

    def soft_update(self, net, target_net):   ## EMA的方式update目标网络的，每次update很少的内容
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        ## 拿到这条序列内的 状态、动作和奖励，下一个状态、是否完成的
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        ## 下个状态+下个状态的动作，目标状态动作价值网络 输出（状态，动作）对的动作价值 Q 
        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        ## 用下一个（状态、动作）对的动作价值 Q，然后间接求出当前的（状态、动作）对的动作价值
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        ## 直接求出当前（状态、动作）的动作价值，和 间接求出的动作价值，使用 MSE 来算损失函数的，q_targets不反向传播求梯度
        critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets))
        self.critic_optimizer.zero_grad()       ## 价值网络的参数梯度置零的
        critic_loss.backward()                  ## 价值网络的损失loss反向传播梯度
        self.critic_optimizer.step()            ## update 网络

        ## update以后的动作价值网络，使用当前的（状态、动作）的动作价值，加了负号的，也就是最大化动作价值的
        ## self.actor(states) 是输出当前状态的动作
        '''
         确定性策略网络的 update，稍稍复杂些，嵌套了两个网络，首先用当前的状态和策略网络，输出当前状态的动作predict_A=self.actor(states)，
         然后 self.critic(states, self.actor(states))输出了当前（状态、动作）的动作价值Q，并求均值加负号，也就是最大化动作价值，
         反向传播用来update 策略网络actor，因critic动作价值网络上面已经update过了，所以下面就只update策略网络actor，critic网络不变的。
        '''
        a = self.actor(states)
        c = self.critic(states, a)
        actor_loss = -torch.mean(c)
        kl_grad = torch.autograd.grad(actor_loss, c, create_graph=True, retain_graph=True)
        kl_a = torch.autograd.grad(actor_loss, a, create_graph=True, retain_graph=True)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        ## 延迟少量的update网络，也就是EMA的方式
        self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
        self.soft_update(self.critic, self.target_critic)  # 软更新价值网络

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
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = 'Pendulum-v1'
env = gym.make(env_name, render_mode="rgb_array")
random.seed(0)
np.random.seed(0)
_ = env.reset(seed=0)
torch.manual_seed(0)
replay_buffer = rl_utils.ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]  # 动作最大值
agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device)

return_list = rl_utils.train_off_policy_agent_withpth(env, agent, num_episodes, replay_buffer, minimal_size, batch_size, 6, r'C:\Users\10696\Desktop\access\Hands-on-RL', '13')

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