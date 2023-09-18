# https://github.com/boyu-ai/Hands-on-RL
# https://github.com/ZouJiu1/Hands-on-RL

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
import numpy as np
import collections
from tqdm import tqdm
# import torch
from net.fullconnect import fclayer
from net.activation import ReLU
from net.loss import mean_square_loss
# import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils
import cv2
import imageio

## 保存历史数据，采样历史数据，拿到历史数据
class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity): ## 容量的大小
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))  ## 加入到队列内部

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)  ## 随机采样的呢，拿到采样的历史数据
        state, action, reward, next_state, done = zip(*transitions)  ## 使用zip来转置，也就是不同的自变量在不同的行
        return np.array(state), action, reward, np.array(next_state), done  ## 状态、动作、奖励、下一个状态，是否结束的

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)  ##  返回历史数据的总长度

## 构造智能体agent的大脑，也就是输入状态，返回该状态下，每个动作的动作价值
## 输入是状态的，也就是 (车子center-point的坐标，车子的速度，杆的竖直角度，杆的角速度)
## 返回值应该是2 dim，
class Qnet():
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim, adam = False):
        super(Qnet, self).__init__()
        self.fc1 = fclayer(state_dim, hidden_dim, adam=adam)
        self.fc2 = fclayer(hidden_dim, action_dim, adam=adam)
        self.relu = ReLU()

    def forward(self, x):
        self.x = x
        self.x1 = self.fc1.forward(x)
        self.x2 = self.relu.forward(self.x1)  # 隐藏层使用ReLU激活函数
        self.x3 = self.fc2.forward(self.x2)
        return self.x3
    
    def backward(self, delta):
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

def train_single():
    inputs = np.random.rand(60, 4)
    outputs = np.random.rand(60, 2)
    state_dim = 2*2
    hidden_dim = 200
    action_dim = 2
    adam = False
    
    fc = Qnet(state_dim, hidden_dim, action_dim, adam)
    for i in range(100000):
        out = fc.forward(inputs)
        sum = np.sum((outputs - out) * (outputs - out))
        delta = 2*(out - outputs)
        partial = fc.backward(delta)
        fc.update(0.0001)
        fc.setzero()
        print(sum)

# train_single()

##  构造智能体的
class DQN:
    ''' DQN算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update):
        self.action_dim = action_dim  ##  动作的dim
        ## 实例化智能体的大脑
        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim, adam=False)  # Q网络
        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim, adam=False)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.learning_rate = learning_rate

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim) ## 随机选择一个动作
        else:
            state = np.array([state])                   ## 状态
            ret = self.q_net.forward(state)
            action = np.argmax(ret)              ## 智能体的大脑，根据状态拿到动作价值，q_net返回值是每个动作的动作价值，然后拿到动作的
        self.epsilon = self.epsilon * 0.996 if self.epsilon > 0.0001 else 0.0001
        return action

    ## 使用历史数据来训练智能体的大脑，两个大脑的，一个实时update做label，另一个延迟update做predict
    '''
    损失函数需要label和predict，但是强化学习是没有label的，所以需要人工构造label才行，DQN使用的构造方式是使用update步长不相同的两个Qnet，
    update数量较多的Qnet来做label，数量较少的Qnet来做predict。所以label就是使用的newest update的Qnet。相当是两个人，一个学的时间长些，
    一个学的时间短些，所以这个学习时间更长的人，可以当老师了，学习时间短的人，就只能当学生了。老师可以教导学生的。所以两个大脑模型Qnet，
    一个做老师，一个是学生，也就是一个label另一个则是predict。这样就构造出了相应的label。
    target_q_net使用下一个状态，然后Q-learning来predict当前(状态，动作)的动作价值，predict
    q_net使用当前状态和动作，直接算出当前(状态，动作)的动作价值，label
    '''
    def update(self, transition_dict):
        ## 初始化 状态、动作、奖励、下一个状态、是否结束的
        states = np.array(transition_dict['states'])
        actions = np.array(transition_dict['actions']).reshape((-1, 1))
        rewards = np.array(transition_dict['rewards']).reshape((-1, 1))
        next_states = np.array(transition_dict['next_states'])
        dones = np.array(transition_dict['dones']).reshape((-1, 1))

        ## target, 因该网络实时update，所以可以看作是真实值，也就是监督学习内的label，而目标网络延迟很多，目标网络的输出可以看作predict
        ## “真实”label，q_net输入当前的状态，返回值是当前状态下每个动作的动作价值，所以gather以后拿到的是：当前(状态和动作)对应的动作价值
        q_net_ret = self.q_net.forward(states)  # Q值
        q_values = []
        loss_delta = np.zeros_like(q_net_ret, dtype=np.float32)
        for i in range(len(q_net_ret)):
            # k = actions[i][0]
            # kk = q_net_ret[i, k]
            q_values.append(q_net_ret[i, actions[i]])
        q_values = np.array(q_values)
        # 下个状态的最大Q值，延迟网络的输出
        ## 可以看作是predict，输入是下一个状态，返回值是下一个状态所有动作的动作价值内的最大值
        ## 用来算当前(状态和动作)下的动作价值
        target_q_net_rett = self.target_q_net.forward(next_states)
        max_next_q_values = []
        
        for i in range(len(target_q_net_rett)):
            kk = max(target_q_net_rett[i])
            max_next_q_values.append(kk)
        max_next_q_values = np.array(max_next_q_values).reshape((-1, 1))
        ## Q-learning algorithm，算出的是当前(状态和动作)的动作价值
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        ## q_net的truth label 和 target_q_net的predict，算损失用来反向传播
        ## 也就是两个网络算出来的（状态和动作）对应的动作价值，用MSE来算损失函数的呢
        dqn_loss, delta = mean_square_loss(q_values, q_targets)  # 均方误差损失函数
        for i in range(len(loss_delta)):
            loss_delta[i, actions[i]] = delta[i]
        self.q_net.backward(loss_delta)   ##  反向传播求出梯度
        self.q_net.update(self.learning_rate) ##  使用累加的梯度来update参数
        self.q_net.setzero()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0

        if self.count % self.target_update == 0:    ##  达到了给定的步长，复制最newest的参数给target_q_net网络，q_net是label，target_q_net是predict
            self.target_q_net.setparam(self.q_net) ## update目标网络
        self.count += 1

lr = 0.02
num_episodes = 500
hidden_dim = 300
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 100000
minimal_size = 500
batch_size = 64

env_name = 'CartPole-v1'
env = gym.make(env_name, render_mode="rgb_array")
_ = env.reset(seed=0)
random.seed(0)
np.random.seed(0)
gym.utils.seeding.np_random(0)
replay_buffer = ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update)

return_list = []
allimage = []
epoch = 10
saved = False
for i in range(epoch):
    ## 训练的次数是
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0   ## 累积的奖励
            state = env.reset()  ## 环境随机重置的
            
            if len(state)!=2*2:
                state = state[0]
            done = False
            # https://huggingface.co/learn/deep-rl-course/unit4/hands-on#create-a-virtual-display
            while not done:
                if (i_episode + 1) % 10 == 0 and i == epoch - 1: # np.mean(return_list[-6:]) > 600 - 101:
                    img = env.render()
                    allimage.append(img)
                    saved = True
                # cv2.imshow("CartPole-v1", img)
                # cv2.waitKey(-1)
                
                action = agent.take_action(state) ## 拿到动作价值最大的动作，取值可选值是：0 或者 1
                ## 环境根据动作，前进一步的，拿到下一个状态，奖励，是否终止，是否步长太长，info
                next_state, reward, terminated, truncated, info = env.step(action)  
                done = terminated | truncated  ## 终止或者步长太长，都会导致已经结束
                ## 将状态、动作、奖励、下一个状态、是否结束，加入到缓冲池，也就是历史内，用来训练大脑网络的
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state   ## 下一个状态赋值到当前状态
                episode_return += reward  ##累加奖励的
                # 当buffer数据的数量超过一定值后,才进行Q网络训练
                if replay_buffer.size() > minimal_size:         ## 不停的和环境交互，直到缓冲池内的历史数据大于一定的数量，再开始训练网络的
                    ## 从缓冲池采样历史数据，用来训练大脑网络的
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {      ##  拿到的历史数据
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    agent.update(transition_dict)  ## 训练大脑网络 q_net
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)
    if saved:
        break

#https://github.com/guicalare/Img2gif/blob/master/Code/Img2Gif.py
imageio.mimsave(r'C:\Users\10696\Desktop\access\Hands-on-RL\numpy_RL_reinforcement_learning\chapter7_DQN.gif', allimage, duration=10)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()
plt.close()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()
plt.close()