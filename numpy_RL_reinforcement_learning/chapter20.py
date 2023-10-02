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
import numpy as np
from tqdm import tqdm
from tqdm import tqdm
import imageio
import sys
import rl_utils
sys.path.append(r'C:\Users\10696\Desktop\access\ma-gym')
from ma_gym.envs.combat.combat import Combat
import matplotlib.pyplot as plt
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
        self.fc2 = fclayer(hidden_dim, hidden_dim, adam=adam)
        self.fc3 = fclayer(hidden_dim, action_dim, adam=adam)
        self.softmax = Softmax()
        self.relu0 = ReLU()
        self.relu1 = ReLU()

    def forward(self, x):
        self.x = x
        self.x1 = self.fc1.forward(x)
        self.x2 = self.relu0.forward(self.x1)  # 隐藏层使用ReLU激活函数
        self.x3 = self.fc2.forward(self.x2)
        self.x6 = self.relu1.forward(self.x3)  # 隐藏层使用ReLU激活函数
        self.x9 = self.fc3.forward(self.x6)
        self.x00 = self.softmax.forward(self.x9, axis=-1)
        return self.x00
    
    def backward(self, delta):
        delta = self.softmax.backward(delta, self.x00)
        delta = self.fc3.backward(delta, inputs=self.x6)
        delta = self.relu1.backward(delta)
        delta = self.fc2.backward(delta, inputs=self.x2)
        delta = self.relu0.backward(delta)
        delta = self.fc1.backward(delta, inputs=self.x)
        return delta
    
    def update(self, lr):
        self.fc3.update(lr)
        self.fc2.update(lr)
        self.fc1.update(lr)
    
    def setzero(self):
        self.fc3.setzero()
        self.fc2.setzero()
        self.fc1.setzero()

    def save_model(self):
        return [self.fc1.save_model(), self.fc2.save_model(), self.fc3.save_model()]
        
    def restore_model(self, models):
        self.fc1.restore_model(models[0])
        self.fc2.restore_model(models[1])
        self.fc3.restore_model(models[2])


## 构造智能体 agent 的大脑，也就是输入状态，返回该状态下，每个动作的动作价值
## 输入是状态的，也就是 (车子center-point的坐标，车子的速度，杆的竖直角度，杆的角速度)
## 返回值应该是2 dim
class ValueNet():
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, adam = True):
        super(ValueNet, self).__init__()
        self.fc1 = fclayer(state_dim, hidden_dim, adam=adam)
        self.fc2 = fclayer(hidden_dim, hidden_dim, adam=adam)
        self.fc3 = fclayer(hidden_dim, 1, adam=adam)
        self.relu0 = ReLU()
        self.relu1 = ReLU()

    def forward(self, x):
        self.x = x
        self.x1 = self.fc1.forward(x)
        self.x2 = self.relu0.forward(self.x1)  # 隐藏层使用ReLU激活函数
        self.x3 = self.fc2.forward(self.x2)
        self.x6 = self.relu1.forward(self.x3)  # 隐藏层使用ReLU激活函数
        self.x9 = self.fc3.forward(self.x6)
        return self.x9

    def backward(self, delta):
        delta = self.fc3.backward(delta, inputs = self.x6)
        delta = self.relu1.backward(delta)
        delta = self.fc2.backward(delta, inputs = self.x2)
        delta = self.relu0.backward(delta)
        delta = self.fc1.backward(delta, inputs = self.x)
        return delta
    
    def update(self, lr):
        self.fc3.update(lr)
        self.fc2.update(lr)
        self.fc1.update(lr)
    
    def setzero(self):
        self.fc3.setzero()
        self.fc2.setzero()
        self.fc1.setzero()

    def save_model(self):
        return [self.fc1.save_model(), self.fc2.save_model(), self.fc3.save_model()]
        
    def restore_model(self, models):
        self.fc1.restore_model(models[0])
        self.fc2.restore_model(models[1])
        self.fc3.restore_model(models[2])

class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim) ##  策略网络的
        self.critic = ValueNet(state_dim, hidden_dim) ##  价值网络
        self.gamma = gamma   ## 衰减因子的呢
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
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
                    dt0 = (-1 / num / 2.0)  * advantage[i][0]
                    dt1 = (-1 / num / 2.0)  * advantage[i][0]
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

actor_lr = 3e-4
critic_lr = 1e-3
num_episodes = 100000
hidden_dim = 64
gamma = 0.99
lmbda = 0.97
eps = 0.2
epochs = 1

team_size = 2
grid_size = (15, 15)
#创建Combat环境，格子世界的大小为15x15，己方智能体和敌方智能体数量都为2
env = Combat(grid_shape=grid_size, n_agents=team_size, n_opponents=team_size)

state_dim = env.observation_space[0].shape[0]
action_dim = env.action_space[0].n
#两个智能体共享同一个策略
agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, 
            eps, gamma)

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
pth = r'C:\Users\10696\Desktop\access\Hands-on-RL\numpy_RL_reinforcement_learning'
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