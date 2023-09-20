import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils
import imageio
from tqdm import tqdm
from numpy_RL_reinforcement_learning.net.loss import mean_square_two_derive_loss

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
        return F.softmax(self.fc2(x), dim=1)  ## 返回该状态下，选择的动作的概率
    
## 构造智能体 agent 的大脑，也就是输入状态，返回该状态下，每个动作的动作价值
## 输入是状态的，也就是 (车子center-point的坐标，车子的速度，杆的竖直角度，杆的角速度)
## 返回值应该是2 dim
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

## 智能体
class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 gamma, device):
        # 策略网络 Actor
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)  # 价值网络  Critic
        # 策略网络优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)  # 价值网络优化器
        self.gamma = gamma  # 折扣因子
        self.device = device

    def take_action(self, state):            # 根据动作概率分布随机采样
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)       ## 拿到该状态下，每个动作的选择概率
        action_dist = torch.distributions.Categorical(probs)    ##   配置 好采样的概率
        action = action_dist.sample()        ## 对该状态下，所有的动作采样，采样的概率是probs
        return action.item()                 ## 返回依概率采样得到的动作

    ## 训练策略网络的，用一条序列来训练
    def update(self, transition_dict):
        ## 拿到这条序列内的 奖励、状态和动作
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

        # 时序差分目标
        ## 用下一个状态，critic求出下个状态的动作价值，然后求出当前状态的动作价值
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        nowval = self.critic(states) ## critic求当前状态的状态动作价值
        ## critic使用当前状态，求出当前状态的动作价值，两者的差就是差分
        td_delta = td_target - nowval  # 时序差分误差
        # log_probs = torch.log(self.actor(states).gather(1, actions))     ## 选择的动作的动作概率，并求 log
        # ## 策略网络的损失，差分越小越好，-log_probs > 0
        # actor_loss = torch.mean(-log_probs * td_delta.detach())          ## 时序差分误差，乘上相应的 log值，就得到策略网络的损失loss
        
        
        
        nowact = self.actor(states)
        q_probably = []
        for ik in range(len(nowact)):
            q_probably.append(nowact[ik, actions[ik][0]])
        q_probably = torch.unsqueeze(torch.tensor(q_probably), -1)
        log_probs = torch.log(q_probably)     ## 选择的动作的动作概率，并求 log
        ## 策略网络的损失，差分越小越好，-log_probs > 0
        ## -log_probs > 0, 所以越靠近0越好，当q_probably=1时最小，也就是选择的动作概率越大越好, td_delta越来越小，log_probs要越来越大才行
        actor_loss = torch.mean(-log_probs * td_delta.detach())          ## 时序差分误差，乘上相应的 log值，就得到策略网络的损失loss
        ## 求出相关的梯度，用来反向传播运算
        num = log_probs.numel()
        log_probs_delta = -td_delta.detach() / num
        # td_delta_delta = -log_probs / num   ## td_delta.detach，不需要求梯度的
        q_probably_delta = log_probs_delta * (1 / q_probably)
        now_act_delta = torch.zeros_like(nowact)
        for ik in range(len(nowact)):
            if actions[ik][0] > 0:
                now_act_delta[ik, 1] = q_probably_delta[ik][0]
            else:
                now_act_delta[ik, 0] = q_probably_delta[ik][0]
        # self.actor.setzero()  ## 默认梯度会累积,这里需要显式将梯度置为0
        # self.actor.backward(now_act_delta) ##  反向传播求出 actor 的梯度
        # self.actor.update(lr=self.actor_lr)##  使用累加的梯度来update参数
        ## 反向传播到critic网络，求出梯度的
        # nowval_d = -td_delta_delta ## td_delta.detach，不需要求梯度的
        # critic_next_states_d = td_delta_delta ## td_delta.detach，不需要求梯度的
        def mean_square_loss(predict, label):
            loss = torch.sum(torch.square(predict - label)) / predict.numel()
            partial0 = 2 * (predict - label) / predict.numel()
            partial1 = 2 * (label - predict) / label.numel()
            return loss, partial0, partial1
        critic_loss, nowval_delta, _ = mean_square_loss(nowval, td_target)    ## td_target.detach，不需要求梯度的
        self.actor_optimizer.zero_grad()     ## 参数的梯度置0
        self.critic_optimizer.zero_grad()    ## 参数的梯度置0
        nowval.backward(nowval_delta)  # 计算策略网络的梯度
        nowact.backward(now_act_delta)  # 计算价值网络的梯度
        self.actor_optimizer.step()  # 更新策略网络的参数
        self.critic_optimizer.step()  # 更新价值网络的参数



        
        # ## 均方误差损失函数，价值网络critic求出当前状态的动作价值，以及用下一个状态间接求出当前状态的动作价值，MSE求损失loss
        # ## 价值网络的损失
        # critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
        # self.actor_optimizer.zero_grad()     ## 参数的梯度置0
        # self.critic_optimizer.zero_grad()    ## 参数的梯度置0
        # actor_loss.backward()  # 计算策略网络的梯度
        # critic_loss.backward()  # 计算价值网络的梯度
        # self.actor_optimizer.step()  # 更新策略网络的参数
        # self.critic_optimizer.step()  # 更新价值网络的参数

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
                    state = next_state    ## 下一个状态赋值到当前状态
                    episode_return += reward  ##累加奖励的
                return_list.append(episode_return)  ## 训练策略网络的，用一条序列来训练
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    # https://github.com/guicalare/Img2gif/blob/master/Code/Img2Gif.py
    imageio.mimsave(r'C:\Users\10696\Desktop\access\Hands-on-RL\chapter10.gif', allimage, duration=10)
    return return_list

actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 1000
hidden_dim = 128
gamma = 0.98
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

env_name = 'CartPole-v1'
env = gym.make(env_name, render_mode="rgb_array")
_ = env.reset(seed=0)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                    gamma, device)

return_list = train_on_policy_agent(env, agent, num_episodes)