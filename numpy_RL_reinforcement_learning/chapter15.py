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
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils
from tqdm import tqdm
from net.fullconnect import fclayer
from net.activation import Softmax, ReLU, tanh, sigmoid
from net.loss import mean_square_two_derive_loss, binary_cross_entropy_loss
import os
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
num_episodes = 250
hidden_dim = 128
gamma = 0.98
lmbda = 0.95
epochs = 10
eps = 0.2

env_name = 'CartPole-v1'
env = gym.make(env_name, render_mode="rgb_array")
_ = env.reset(seed=0)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
ppo_agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                epochs, eps, gamma)
## 使用 PPO 来训练专家策略的
return_list = rl_utils.train_on_policy_agent(env, ppo_agent, num_episodes)

## 使用专家策略模型，只采样1个序列，每个序列采样30个状态动作对
def sample_expert_data(n_episode):
    states = []
    actions = []
    for episode in range(n_episode):         ## 1个序列
        state = env.reset()      ## 环境重置的
        if len(state)!=2*2:
            state = state[0]
        done = False  ## 是否已经完成
        while not done:
            action = ppo_agent.take_action(state)  ## 策略根据状态给出动作
            states.append(state)
            actions.append(action)
            ##  环境执行动作，并反馈下一个状态、动作的奖励、是否完成、步长太长的，info
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated       ## 终止或者步长太长，都会导致已经结束
            # next_state, reward, done, _ = env.step(action)   ## 环境执行动作，并给出下一个状态、动作
            state = next_state
    return np.array(states), np.array(actions)  ## 分别返回状态和动作对

_ = env.reset(seed=6)
random.seed(6)
n_episode = 1
expert_s, expert_a = sample_expert_data(n_episode)

n_samples = 30  # 采样30个数据
random_index = random.sample(range(expert_s.shape[0]), n_samples)   ## 从状态动作对内继续采样
expert_s = expert_s[random_index]
expert_a = expert_a[random_index]


## 模仿者智能体
class BehaviorClone:
    def __init__(self, state_dim, hidden_dim, action_dim, lr):
        self.policy = PolicyNet(state_dim, hidden_dim, action_dim)  ## 策略网络的
        self.lr = lr

    ## 使用专家采样数据，模仿者学着靠拢专家策略
    def learn(self, states, actions):
        states = np.array(states)  ## 输入
        actions = np.array(actions).reshape((-1, 1))  ## 输出，也就是监督学习的 truth label
        ## 拿到策略网络输出指定动作的概率的log值
        nowact = self.policy.forward(states)
        imi_probs = []
        for ik in range(len(nowact)):
            imi_probs.append(nowact[ik, actions[ik][0]])
        imi_probs = np.expand_dims(np.array(imi_probs), -1)

        log_probs = np.log(imi_probs)
        ## 最小化 (最大似然估计值的相反数)，也就是 最大化 最大似然估计值
        num = 1
        for i in log_probs.shape:
            num = num * i
        bc_loss = np.mean(-log_probs)  # 最大似然估计
        
        delta = (-1 / num) * (1 / imi_probs)
        now_act_delta = np.zeros_like(nowact)
        for i in range(len(nowact)):
            now_act_delta[i, int(actions[i][0])] = delta[i][0]
            # if actions[i][0] < 1:
            #     now_act_delta[i, 0] = delta[i][0]
            # else:
            #     now_act_delta[i, 1] = delta[i][0]
        
        self.policy.setzero()  ## 默认梯度会累积,这里需要显式将梯度置为0
        self.policy.backward(now_act_delta) ##  反向传播求出 policy 的梯度
        self.policy.update(lr=self.lr)##  使用累加的梯度来update参数

    ## 模仿者采取动作，并按照概率给出采样的动作
    def take_action(self, state):
        state = np.array([state])
        probs = self.policy.forward(state)
        action_dist = torch.distributions.Categorical(torch.tensor(probs))
        action = action_dist.sample()
        return action.item()

def test_agent(agent, env, n_episode, num):
    return_list = []
    allimage = []
    for episode in range(n_episode):
        episode_return = 0
        state = env.reset()
        if len(state)!=2*2:
            state = state[0]
        done = False
        ## 采样一条序列的
        while not done:
            if num==900:
                img = env.render()
                allimage.append(img)
            # cv2.imshow("CartPole-v1"
            action = agent.take_action(state)
            ##  环境执行动作，并反馈下一个状态、动作的奖励、是否完成、步长太长的，info
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated       ## 终止或者步长太长，都会导致已经结束
            # next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_return += reward
        return_list.append(episode_return)
    if num == 900:
        # https://github.com/guicalare/Img2gif/blob/master/Code/Img2Gif.py
        pth = r'C:\Users\10696\Desktop\access\Hands-on-RL\numpy_RL_reinforcement_learning'
        imageio.mimsave(os.path.join(pth, 'chapter%s.gif'%str(15)), allimage, duration=10)
    return np.mean(return_list)

_ = env.reset(seed=6)
torch.manual_seed(6)
np.random.seed(6)

lr = 1e-3
bc_agent = BehaviorClone(state_dim, hidden_dim, action_dim, lr)
n_iterations = 1000
batch_size = 64
test_returns = []

## 行为克隆的
with tqdm(total=n_iterations, desc="进度条") as pbar:
    for i in range(n_iterations):
        sample_indices = np.random.randint(low=0,
                                           high=expert_s.shape[0],
                                           size=batch_size)
        bc_agent.learn(expert_s[sample_indices], expert_a[sample_indices]) ## 模仿者使用专家数据来train
        current_return = test_agent(bc_agent, env, 5, i) ## 模仿者测验结果
        test_returns.append(current_return)
        if (i + 1) % 10 == 0:
            pbar.set_postfix({'return': '%.3f' % np.mean(test_returns[-10:])})
        pbar.update(1)

# 进度条: 100%|██████████| 1000/1000 [00:50<00:00, 19.82it/s, return=42.000]

iteration_list = list(range(len(test_returns)))
plt.plot(iteration_list, test_returns)
plt.xlabel('Iterations')
plt.ylabel('Returns')
plt.title('BC on {}'.format(env_name))
plt.show()


## GAIL的生成器是策略网络，上面已经给出了
## D是判别器的，也就是用来判定给定的（状态，动作）对，目标是专家策略输出0，模仿者策略输出1，一般来说模仿者策略输出越小越好
class Discriminator():
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim, adam = True):
        super(Discriminator, self).__init__()
        self.fc1 = fclayer(state_dim + action_dim, hidden_dim, adam=adam)
        self.fc2 = fclayer(hidden_dim, 1, adam=adam)
        self.relu = ReLU()
        self.sigmoid = sigmoid()

    def forward(self, x, a):
        self.inp = np.concatenate([x, a], axis = 1)
        self.x1  = self.fc1.forward(self.inp)
        self.x2  = self.relu.forward(self.x1)  # 隐藏层使用ReLU激活函数
        self.x3  = self.fc2.forward(self.x2)
        self.x6  = self.sigmoid.forward(self.x3)
        return self.x6   ## 目标专家策略输出0，模仿者策略输出1，还可以用来做奖励函数
    
    def backward(self, delta):
        delta = self.sigmoid.backward(delta)
        delta = self.fc2.backward(delta, inputs=self.x2)
        delta = self.relu.backward(delta)
        delta = self.fc1.backward(delta, inputs=self.inp)
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
    
class GAIL:
    def __init__(self, agent, state_dim, action_dim, hidden_dim, lr_d):
        self.discriminator = Discriminator(state_dim, hidden_dim,
                                           action_dim)  ## 判别器
        self.lr_d = lr_d
        self.agent = agent  ## 模仿者智能体

    ## 网络的训练
    def learn(self, expert_s, expert_a, agent_s, agent_a, next_s, dones):
        ## 初始化需要的变量
        expert_states = np.array(expert_s)  ## 专家输入的状态
        expert_actions = np.array(expert_a)    ## 专家输出的动作
        agent_states = np.array(agent_s)  ##  模仿者输入状态
        agent_actions = np.array(agent_a)  ## 模仿者输入动作
        expert_actions = F.one_hot(torch.tensor(expert_actions).long(), num_classes=2).float().cpu().numpy()   ## 专家动作来 one-hot 标签化
        agent_actions = F.one_hot(torch.tensor(agent_actions).long(), num_classes=2).float().cpu().numpy()  ##  模仿者动作 one-hot 标签化

        self.discriminator.setzero()  ## 默认梯度会累积,这里需要显式将梯度置为0
        ## D判别器给出专家（状态，动作）对的概率，目标是靠拢 0
        expert_prob = self.discriminator.forward(expert_states, expert_actions)
        losse, partiale, sigmoide = binary_cross_entropy_loss(expert_prob, np.zeros_like(expert_prob))
        self.discriminator.backward(partiale) ##  反向传播求出 discriminator 的梯度
        ## D判别器给出模仿者（状态，动作）对的概率，目标是靠拢 1
        agent_prob = self.discriminator.forward(agent_states, agent_actions) 
        ## D判别器的目标是，专家的靠拢 0，模仿者靠拢 1
        lossa, partiala, sigmoida = binary_cross_entropy_loss(agent_prob, np.ones_like(agent_prob))
        self.discriminator.backward(partiala) ##  反向传播求出 discriminator 的梯度
        # delta = partiala + partiale
        self.discriminator.update(lr=self.lr_d) ##  使用累加的梯度来update参数

        ## 损失函数来做奖励，D判别器给出模仿者的概率，越小越好的，专家的才会越小，说明此时判别器已经误判了，不能准确区分专家和模仿者的策略。
        rewards = -np.log(agent_prob)
        transition_dict = {
            'states': agent_s,
            'actions': agent_a,
            'rewards': rewards,
            'next_states': next_s,
            'dones': dones
        }
        self.agent.update(transition_dict)  ## 模仿者 train 策略网络和价值网络，此处的强化学习算法可以使用其他的

_ = env.reset(seed=0)
torch.manual_seed(0)
lr_d = 1e-3
## 模仿者智能体
agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
            epochs, eps, gamma)
## 生成式对抗模仿
gail = GAIL(agent, state_dim, action_dim, hidden_dim, lr_d)
n_episode = 500
return_list = []

## 生成式对抗模仿学习
with tqdm(total=n_episode, desc="进度条") as pbar:
    allimage = []
    for i in range(n_episode):
        episode_return = 0
        state = env.reset()
        if len(state)!=2*2:
            state = state[0]
        done = False
        state_list = []
        action_list = []
        next_state_list = []
        done_list = []
        while not done:
            if i==n_episode - 1:
                img = env.render()
                allimage.append(img)
            # cv2.imshow("CartPole-v1"
            action = agent.take_action(state)
            ##  环境执行动作，并反馈下一个状态、动作的奖励、是否完成、步长太长的，info
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated       ## 终止或者步长太长，都会导致已经结束
            # next_state, reward, done, _ = env.step(action)
            state_list.append(state)
            action_list.append(action)
            next_state_list.append(next_state)
            done_list.append(done)
            state = next_state
            episode_return += reward
        return_list.append(episode_return)
        ## 生成式对抗模仿学习
        ## 使用了专家策略模型采样的数据 expert_s, expert_a
        ## 以及模仿者智能体 采样的数据 state_list，action_list，next_state_list, done_list来train
        gail.learn(expert_s, expert_a, state_list, action_list,
                   next_state_list, done_list)
        if (i + 1) % 10 == 0:
            pbar.set_postfix({'return': '%.3f' % np.mean(return_list[-10:])})
        pbar.update(1)
        if i==n_episode - 1:
            # https://github.com/guicalare/Img2gif/blob/master/Code/Img2Gif.py
            pth = r'C:\Users\10696\Desktop\access\Hands-on-RL\numpy_RL_reinforcement_learning'
            imageio.mimsave(os.path.join(pth, 'chapter%s.gif'%str('15GAIL')), allimage, duration=10)
# 进度条: 100%|██████████| 500/500 [04:08<00:00,  2.01it/s, return=200.000]

iteration_list = list(range(len(return_list)))
plt.plot(iteration_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('GAIL on {}'.format(env_name))
plt.show()