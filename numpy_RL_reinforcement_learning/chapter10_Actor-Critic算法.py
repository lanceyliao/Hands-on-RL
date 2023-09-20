import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import imageio
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
    
    def setparam(self, model):
        self.fc1.params = model.fc1.params.copy()
        self.fc2.params = model.fc2.params.copy()
        if self.fc1.bias:
            self.fc1.bias_params = model.fc1.bias_params.copy()
        if self.fc2.bias:
            self.fc2.bias_params = model.fc2.bias_params.copy()
    
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

    def forward(self, x, saveparam = None):
        self.x = x
        self.x1 = self.fc1.forward(x)
        self.x2 = self.relu.forward(self.x1)  # 隐藏层使用ReLU激活函数
        self.x3 = self.fc2.forward(self.x2)
        if saveparam!=None:
            self.param[saveparam] = [self.x.copy(), self.x2.copy()]
        else:
            exit(-1)
        return self.x3

    def backward(self, delta, saveparam=None):
        if saveparam!=None:
            delta = self.fc2.backward(delta, inputs = self.param[saveparam][1])
            delta = self.relu.backward(delta)
            delta = self.fc1.backward(delta, inputs = self.param[saveparam][0])
            self.param.pop(saveparam)
        else:
            exit(-1)
            # delta = self.fc2.backward(delta, inputs=self.x2)
            # delta = self.relu.backward(delta)
            # delta = self.fc1.backward(delta, inputs=self.x)
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

## 智能体
class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma):
        # 策略网络 Actor
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, adam = True)
        self.critic = ValueNet(state_dim, hidden_dim, adam = True)  # 价值网络  Critic
        self.gamma = gamma  # 折扣因子
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

    def take_action(self, state):            # 根据动作概率分布随机采样
        state = np.array([state])
        probs = self.actor.forward(state)       ## 拿到该状态下，每个动作的选择概率
        action_dist = torch.distributions.Categorical(torch.tensor(probs))    ##   配置 好采样的概率
        action = action_dist.sample()        ## 对该状态下，所有的动作采样，采样的概率是probs
        return action.item()                 ## 返回依概率采样得到的动作

    ## 训练策略网络的，用一条序列来训练
    ## 不用遍历了的，可以批量来处理，因不需要求每个状态的回报，价值函数迭代不需要遍历，直接求就可以的呢
    def update(self, transition_dict):
        ## 拿到这条序列内的 奖励、状态和动作
        states = np.array(transition_dict['states'])
        actions = np.array(transition_dict['actions']).reshape((-1, 1))
        rewards = np.array(transition_dict['rewards']).reshape((-1, 1))
        next_states = np.array(transition_dict['next_states'])
        dones = np.array(transition_dict['dones']).reshape((-1, 1))

        # 时序差分目标
        ##    用下一个状态，critic求出下个状态的动作价值，然后求出当前状态的动作价值
        td_target = rewards + self.gamma * self.critic.forward(next_states, 0) * (1 - dones)        ## 真实标签的，truth label，有监督
        nowval = self.critic.forward(states, 6)        ## critic求当前状态的状态动作价值，predict值
        ## critic使用当前状态，求出当前状态的动作价值，两者的差就是差分
        td_delta = td_target - nowval  # 时序差分误差

        nowact = self.actor.forward(states)
        q_probably = []
        for ik in range(len(nowact)):
            q_probably.append(nowact[ik, actions[ik][0]])
        q_probably = np.expand_dims(np.array(q_probably), -1)
        log_probs = np.log(q_probably)     ## 选择的动作的动作概率，并求 log
        ## 策略网络的损失，差分越小越好，-log_probs > 0，td_delta.detach()也就是不用反向求梯度，这的td_delta看作是固定的值
        ## -log_probs > 0, 所以越靠近0越好，当q_probably=1时最小，也就是选择的动作概率越大越好, td_delta越来越小，log_probs要越来越大才行
        actor_loss = np.mean(-log_probs * td_delta)              ## 时序差分误差，乘上相应的 log值，就得到策略网络的损失loss
        
        ## 求出相关的梯度，用来反向传播运算
        num = log_probs.size
        log_probs_delta = -td_delta / num
        # td_delta_delta = -log_probs / num                      ## td_delta.detach，不需要求梯度的
        q_probably_delta = log_probs_delta * (1 / q_probably)
        now_act_delta = np.zeros_like(nowact)
        for ik in range(len(nowact)):
            if actions[ik][0] > 0:
                now_act_delta[ik, 1] = q_probably_delta[ik][0]
            else:
                now_act_delta[ik, 0] = q_probably_delta[ik][0]
        self.actor.setzero()  ## 默认梯度会累积,这里需要显式将梯度置为0
        self.actor.backward(now_act_delta) ##  反向传播求出 actor 的梯度
        self.actor.update(lr=self.actor_lr)##  使用累加的梯度来update参数
        ## 反向传播到critic网络，求出梯度的    ## td_delta.detach，不需要求梯度的
        # nowval_d = -td_delta_delta    ## td_delta.detach，不需要求梯度的
        # critic_next_states_d = td_delta_delta   ## td_delta.detach，不需要求梯度的
        
        ## 均方误差损失函数，价值网络critic求出当前状态的动作价值，以及用下一个状态间接求出当前状态的动作价值，MSE求损失loss
        ## 价值网络的损失，td_target.detach()不用反向求梯度，所以td_target看作truth label，nowval看作predict
        critic_loss, nowval_delta, _ = mean_square_two_derive_loss(nowval, td_target)    ## td_target.detach，不需要求梯度的
        self.critic.setzero()  ## 默认梯度会累积,这里需要显式将梯度置为0
        self.critic.backward(nowval_delta, 6)  ##  反向传播求出 critic 的梯度，使用保存的参数6算梯度
        self.critic.update(self.critic_lr) ##  使用累加的梯度来update参数

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    allimage = []
    epoch = 10
    limit = 2000
    for i in range(epoch):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                if len(state)!=2*2:
                    state = state[0]
                done = False
                ## 采样一条序列的
                while not done:
                    if (i_episode + 1) % 10 == 0 and i == epoch - 1 and len(allimage) < limit:
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
                    state = next_state              ## 下一个状态赋值到当前状态
                    episode_return += reward        ## 累加奖励的
                return_list.append(episode_return)  ## 训练策略网络的，用一条序列来训练
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    # https://github.com/guicalare/Img2gif/blob/master/Code/Img2Gif.py
    imageio.mimsave(r'C:\Users\10696\Desktop\access\Hands-on-RL\numpy_RL_reinforcement_learning\chapter10.gif', allimage, duration=10)
    return return_list

actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 1000
hidden_dim = 128
gamma = 0.98

env_name = 'CartPole-v1'
env = gym.make(env_name, render_mode="rgb_array")
_ = env.reset(seed=0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma)

return_list = train_on_policy_agent(env, agent, num_episodes)