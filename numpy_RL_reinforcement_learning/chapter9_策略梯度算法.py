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

import gymnasium as gym
import numpy as np
from net.fullconnect import fclayer
from net.activation import Softmax, ReLU
import matplotlib.pyplot as plt
import rl_utils
from tqdm import tqdm
import torch
import imageio

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
    outputs[outputs < 0] = 0
    state_dim = 2*2
    hidden_dim = 200
    action_dim = 2
    adam = True
    
    fc = PolicyNet(state_dim, hidden_dim, action_dim, adam)
    for i in range(100000):
        out = fc.forward(inputs)
        sum = np.sum((outputs - out) * (outputs - out))
        delta = 2*(out - outputs)
        partial = fc.backward(delta)
        fc.update(0.0001)
        fc.setzero()
        print(sum)

# train_single()

## 策略梯度算法
class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma):
        ## 实例化智能体的大脑
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim, adam=True)
        self.gamma = gamma  # 折扣因子
        self.learning_rate = learning_rate

    def take_action(self, state):            # 根据动作概率分布随机采样
        state = np.array([state])
        probs = self.policy_net.forward(state)       ## 拿到该状态下，每个动作的选择概率
        action_dist = torch.distributions.Categorical(torch.tensor(probs))    ##   配置 好采样的概率
        action = action_dist.sample()        ## 对该状态下，所有的动作采样，采样的概率是probs
        return action.item()                 ## 返回依概率采样得到的动作

    ## 训练策略网络的，用一条序列来训练
    def update(self, transition_dict):
        ## 拿到这条序列内的 奖励、状态和动作
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0      ##   初始化回报值 = 0
        self.policy_net.setzero()  ## 默认梯度会累积,这里需要显式将梯度置为0
        for i in reversed(range(len(reward_list))):  # 从最后一步算起
            reward = reward_list[i]   ##  拿到这一步的奖励
            state = np.array([state_list[i]])    ##  拿到这一步的状态
            action = np.array([action_list[i]]).reshape((-1, 1))         ##  拿到这一步的动作

            ret = self.policy_net.forward(state)
            q_probably = []
            for ik in range(len(ret)):
                q_probably.append(ret[ik, action[ik][0]])
            q_probably = np.array(q_probably)
            
            log_prob = np.log(q_probably)          ## 算动作概率的log值 
            G = self.gamma * G + reward   ## 算这一步状态s的回报
            loss = -log_prob * G  # 每一步的损失函数             ##  算这一步的动作回报，并取相反数的
            # 反向传播计算梯度 ## 梯度会累加的
            grad_log_prob = -G
            grad_q_probably = grad_log_prob * (1.0 / q_probably)
            grad_rett = [0.0, grad_q_probably[0]] if action[0][0] > 0 else [grad_q_probably[0], 0.0] 
            grad_rett = np.array([grad_rett])
            self.policy_net.backward(grad_rett)   ##  反向传播求出梯度

        self.policy_net.update(self.learning_rate) ##  使用累加的梯度来update参数
        # self.optimizer.step()  # 梯度下降 update 参数 ## 所以一个序列以后，网络 policy_net 的参数才会 update

learning_rate = 1e-3
num_episodes = 1000
hidden_dim = 300
gamma = 0.98

env_name = "CartPole-v1"
env = gym.make(env_name, render_mode="rgb_array")
_ = env.reset(seed=0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = REINFORCE(state_dim, hidden_dim, action_dim, learning_rate, gamma)

return_list = []
allimage = []
epoch = 10
for i in range(epoch):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            transition_dict = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': []
            }
            state = env.reset()
            if len(state)!=2*2:
                state = state[0]
            done = False
            # https://huggingface.co/learn/deep-rl-course/unit4/hands-on#create-a-virtual-display
            ## 采样一条序列的
            while not done:
                if (i_episode + 1) % 10 == 0 and i == epoch - 1:
                    img = env.render()
                    allimage.append(img)
                # cv2.imshow("CartPole-v1", img)
                # cv2.waitKey(-1)
                action = agent.take_action(state)    ##  根据状态采取动作的
                ##  环境执行动作，并反馈下一个状态、动作的奖励、是否完成、步长太长的，info
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated | truncated       ## 终止或者步长太长，都会导致已经结束
                ## record该序列的 该时刻状态、该时刻动作、下一个状态、动作的奖励、是否完成的
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state    ## 下一个状态赋值到当前状态
                episode_return += reward  ##累加奖励的
            return_list.append(episode_return)
            agent.update(transition_dict)  ## 训练策略网络的，用一条序列来训练
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)

# https://github.com/guicalare/Img2gif/blob/master/Code/Img2Gif.py
imageio.mimsave(r'C:\Users\10696\Desktop\access\Hands-on-RL\numpy_RL_reinforcement_learning\chapter9.gif', allimage, duration=10)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('REINFORCE on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('REINFORCE on {}'.format(env_name))
plt.show()