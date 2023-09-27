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
from tqdm import tqdm
import imageio

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

## 首先使用强化学习算法 PPO 来训练专家策略，然后用来采样的，产生专家数据
class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)  ##  策略网络的
        self.critic = ValueNet(state_dim, hidden_dim).to(device)  ##  价值网络
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)  ##  函数配置优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)  ##  价值函数配置优化器
        self.gamma = gamma   ## 衰减因子的呢
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用于训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state):            # 根据动作概率分布随机采样
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)       ## 拿到该状态下，每个动作的选择概率
        action_dist = torch.distributions.Categorical(probs)    ##   配置 好采样的概率
        action = action_dist.sample()        ## 对该状态下，所有的动作采样，采样的概率是probs
        return action.item()                 ## 返回依概率采样得到的动作

    def update(self, transition_dict):
        ## 拿到这条序列内的 奖励、状态和动作，下一个状态、是否完成的
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        ## 用下个状态求下一个状态的状态动作价值，然后间接求出当前状态的状态动作价值
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        ## 间接求出的价值 - 直接求出的当前状态的状态动作价值，也就是 TD-error，或者是优势函数 A
        td_delta = td_target - self.critic(states)
        ##  算出优势函数，广义优势估计，也就是每一步优势的均值
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        ## 选择的旧动作概率的log值，不反向传播求梯度，detach
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
        for _ in range(self.epochs):
            ## 选择的动作概率的log值，不反向传播求梯度，detach
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)   ## 算重要性采样
            surr1 = ratio * advantage ## 重要性采样和优势估计相乘的
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            ## 直接求出当前状态的状态动作价值，和 间接求出的价值，使用 MSE 来算损失函数的，td_target不反向传播求梯度，detach
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()  ## 价值网络的参数梯度置零的
            actor_loss.backward()
            critic_loss.backward()   ## 价值网络的损失loss反向传播梯度
            self.actor_optimizer.step()
            self.critic_optimizer.step()   # 更新价值函数

actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 250
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
ppo_agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                epochs, eps, gamma, device)
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
        self.policy = PolicyNet(state_dim, hidden_dim, action_dim).to(device)  ## 策略网络的
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)  ## 优化器

    ## 使用专家采样数据，模仿者学着靠拢专家策略
    def learn(self, states, actions):
        states = torch.tensor(states, dtype=torch.float).to(device)  ## 输入
        actions = torch.tensor(actions).view(-1, 1).long().to(device)  ## 输出，也就是监督学习的 truth label
        ## 拿到策略网络输出指定动作的概率的log值
        log_probs = torch.log(self.policy(states).gather(1, actions))
        ## 最小化 (最大似然估计值的相反数)，也就是 最大化 最大似然估计值
        bc_loss = torch.mean(-log_probs)  # 最大似然估计

        self.optimizer.zero_grad() ## 梯度置0
        bc_loss.backward() ## 反向传播求出参数梯度
        self.optimizer.step()  ##  使用梯度来update参数

    ## 模仿者采取动作，并按照概率给出采样的动作
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(device)
        probs = self.policy(state)
        action_dist = torch.distributions.Categorical(probs)
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
        pth = r'C:\Users\10696\Desktop\access\Hands-on-RL'
        imageio.mimsave(os.path.join(pth, 'chapter%s.gif'%str(15)), allimage, duration=10)
    return np.mean(return_list)

# _ = env.reset(seed=0)
# torch.manual_seed(0)
# np.random.seed(0)

# lr = 1e-3
# bc_agent = BehaviorClone(state_dim, hidden_dim, action_dim, lr)
# n_iterations = 1000
# batch_size = 64
# test_returns = []

# ## 行为克隆的
# with tqdm(total=n_iterations, desc="进度条") as pbar:
#     for i in range(n_iterations):
#         sample_indices = np.random.randint(low=0,
#                                            high=expert_s.shape[0],
#                                            size=batch_size)
#         bc_agent.learn(expert_s[sample_indices], expert_a[sample_indices]) ## 模仿者使用专家数据来train
#         current_return = test_agent(bc_agent, env, 5, i) ## 模仿者测验结果
#         test_returns.append(current_return)
#         if (i + 1) % 10 == 0:
#             pbar.set_postfix({'return': '%.3f' % np.mean(test_returns[-10:])})
#         pbar.update(1)

# # 进度条: 100%|██████████| 1000/1000 [00:50<00:00, 19.82it/s, return=42.000]

# iteration_list = list(range(len(test_returns)))
# plt.plot(iteration_list, test_returns)
# plt.xlabel('Iterations')
# plt.ylabel('Returns')
# plt.title('BC on {}'.format(env_name))
# plt.show()

## GAIL的生成器是策略网络，上面已经给出了
## D是判别器的，也就是用来判定给定的（状态，动作）对，目标是专家策略输出0，模仿者策略输出1，一般来说模仿者策略输出越小越好
class Discriminator(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Discriminator, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        return torch.sigmoid(self.fc2(x))  ## 目标专家策略输出0，模仿者策略输出1，还可以用来做奖励函数
    
class GAIL:
    def __init__(self, agent, state_dim, action_dim, hidden_dim, lr_d):
        self.discriminator = Discriminator(state_dim, hidden_dim,
                                           action_dim).to(device)  ## 判别器
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr_d)  ## 优化器
        self.agent = agent  ## 模仿者智能体

    ## 网络的训练
    def learn(self, expert_s, expert_a, agent_s, agent_a, next_s, dones):
        ## 初始化需要的变量
        expert_states = torch.tensor(expert_s, dtype=torch.float).to(device)  ## 专家输入的状态
        expert_actions = torch.tensor(expert_a).to(device)    ## 专家输出的动作
        agent_states = torch.tensor(agent_s, dtype=torch.float).to(device)  ##  模仿者输入状态
        agent_actions = torch.tensor(agent_a).to(device)  ## 模仿者输入动作
        expert_actions = F.one_hot(expert_actions.long(), num_classes=2).float()   ## 专家动作来 one-hot 标签化
        agent_actions = F.one_hot(agent_actions.long(), num_classes=2).float()  ##  模仿者动作 one-hot 标签化

        ## D判别器给出专家（状态，动作）对的概率，目标是靠拢 0
        expert_prob = self.discriminator(expert_states, expert_actions)
        ## D判别器给出模仿者（状态，动作）对的概率，目标是靠拢 1
        agent_prob = self.discriminator(agent_states, agent_actions) 
        ## D判别器的目标是，专家的靠拢 0，模仿者靠拢 1
        discriminator_loss = nn.BCELoss()(
            agent_prob, torch.ones_like(agent_prob)) + nn.BCELoss()(
                expert_prob, torch.zeros_like(expert_prob))
        self.discriminator_optimizer.zero_grad()  ## 梯度置0
        discriminator_loss.backward()  ## 反向传播梯度
        self.discriminator_optimizer.step()  ## 使用梯度来update参数
        ## 损失函数来做奖励，D判别器给出模仿者的概率，越小越好的，专家的才会越小，说明此时判别器已经误判了，不能准确区分专家和模仿者的策略。
        rewards = -torch.log(agent_prob).detach().cpu().numpy()
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
            epochs, eps, gamma, device)
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
            pth = r'C:\Users\10696\Desktop\access\Hands-on-RL'
            imageio.mimsave(os.path.join(pth, 'chapter%s.gif'%str('15GAIL')), allimage, duration=10)
# 进度条: 100%|██████████| 500/500 [04:08<00:00,  2.01it/s, return=200.000]

iteration_list = list(range(len(return_list)))
plt.plot(iteration_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('GAIL on {}'.format(env_name))
plt.show()