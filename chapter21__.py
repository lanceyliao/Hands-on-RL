import os
# import OpenGL.GL as gl
from pyglet.gl import glPushMatrix, glTranslatef, glRotatef, glScalef, glPopMatrix
# from OpenGL.GL import glPushMatrix, glTranslatef, glRotatef, glScalef, glPopMatrix
# import gymnasium as gym
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import random
import rl_utils
import collections
import itertools
from tqdm import tqdm
from scipy.stats import truncnorm
from torch.distributions import Normal
import imageio
import cv2
import matplotlib.pyplot as plt
# !pip install --upgrade gym==0.10.5 -q    gym==0.26.2
import gymnasium as gym
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

## pyglet-1.3.3

def make_env(scenario_name):
    # 从环境文件脚本中创建环境
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                        scenario.observation)
    return env

def onehot_from_logits(logits, eps=0.01):
    ''' 生成最优动作的独热（one-hot）形式 '''
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    # 生成随机动作,转换成独热形式
    rand_acs = torch.autograd.Variable(torch.eye(logits.shape[1])[[
        np.random.choice(range(logits.shape[1]), size=logits.shape[0])
    ]],
                                       requires_grad=False).to(logits.device)
    # 通过epsilon-贪婪算法来选择用哪个动作
    return torch.stack([
        argmax_acs[i] if r > eps else rand_acs[i]
        for i, r in enumerate(torch.rand(logits.shape[0]))
    ])

def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """从Gumbel(0,1)分布中采样"""
    U = torch.autograd.Variable(tens_type(*shape).uniform_(),
                                requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)  ## Gumbel 分布

def gumbel_softmax_sample(logits, temperature):
    """ 从Gumbel-Softmax分布中采样"""
    ## log值，加上 Gumbel分布的，也就是相当是 做了采样的
    ## 这个采样可以对 logits 求梯度，而且 gumbel部分没有梯度的，也就实现了类别采样的同时，可以反向传播梯度，采样和logits数值基本没有关系了
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(
        logits.device)
    return F.softmax(y / temperature, dim=1)  ## softmax, 用 temperature 来控制 和 均匀分布的相似度

def gumbel_softmax(logits, temperature=1.0):
    """从Gumbel-Softmax分布中采样,并进行离散化"""
    y = gumbel_softmax_sample(logits, temperature)
    y_hard = onehot_from_logits(y)  ## one_hot化，之后会用到，不需要求梯度
    '''
    detach 不求梯度的呢，所以 (y_hard.to(logits.device) - y) 内的变量都不会求梯度
    只有 + y 这的 y 会求出梯度来，只对 + y 求梯度的呢
    也就是不用关注 one_hot 这一步，one_hot 这一步不需要求导
    -y + y = 0 的
    所以 下面这步就是要避免对 one_hot 求导，只对 y 求导的
    '''
    y = (y_hard.to(logits.device) - y).detach() + y
    # 返回一个y_hard的独热量,但是它的梯度是y,我们既能够得到一个与环境交互的离散动作,又可以
    # 正确地反传梯度
    return y

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        # for i in range(len(state)):
        #     if len(state[i][0])!=10:
        #         for j in range(10 - len(state[i][0])):
        #             state[i][0] = np.append(state[i][0], 0)
        # for i in range(len(next_state)):
        #     if len(next_state[i][0])!=10:
        #         for j in range(10 - len(next_state[i][0])):
        #             next_state[i][0] = np.append(next_state[i][0], 0)
        # k = np.array(state)
        # kk = np.array(next_state)
        return state, action, reward, next_state, done 

    def size(self): 
        return len(self.buffer)

## 策略网络和价值网络都是使用的相同的网络结构
class TwoLayerFC(torch.nn.Module):
    def __init__(self, num_in, num_out, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_in, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)       ## 都返回 FULL connect 的数值


class DDPG:
    ''' DDPG算法 '''
    def __init__(self, state_dim, action_dim, critic_input_dim, hidden_dim,
                 actor_lr, critic_lr, device):
        self.actor = TwoLayerFC(state_dim, action_dim, hidden_dim).to(device)   ##  策略网络的
        self.target_actor = TwoLayerFC(state_dim, action_dim, 
                                       hidden_dim).to(device)    ##  目标策略网络，延迟update
        self.critic = TwoLayerFC(critic_input_dim, 1, hidden_dim).to(device)    ## 状态动作价值网络
        self.target_critic = TwoLayerFC(critic_input_dim, 1,
                                        hidden_dim).to(device)                  ##  目标状态动作价值网络，延迟update
        # 初始化目标价值网络并设置和价值网络相同的参数
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化目标策略网络并设置和策略相同的参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), ## 策略优化器的
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)           ## 价值优化器的

    def take_action(self, state, explore = False):
        action = self.actor(state)                ## 拿到策略网络的输出
        if explore:
            action = gumbel_softmax(action)       ## 探索，使用依概率采样，返回离散动作的可导
        else:
            action = onehot_from_logits(action)   ## 通过epsilon-贪婪算法来选择用哪个动作
        return action.detach().cpu().numpy()[0]   ## 返回拿到的动作

    def soft_update(self, net, target_net, tau):     ## EMA的方式update目标网络的，每次update很少的内容
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) +
                                    param.data * tau)

class MADDPG:
    def __init__(self, env, device, actor_lr, critic_lr, hidden_dim,
                 state_dims, action_dims, critic_input_dim, gamma, tau):
        self.agents = []
        for i in range(len(env.agents)):
            self.agents.append(
                DDPG(state_dims[i], action_dims[i], critic_input_dim,
                     hidden_dim, actor_lr, critic_lr, device))
        self.gamma = gamma
        self.tau = tau
        self.critic_criterion = torch.nn.MSELoss()
        self.device = device

    @property   ## https://zoujiu.blog.csdn.net/article/details/132531138
    def policies(self):
        return [agt.actor for agt in self.agents]            ## 返回所有的策略网络

    @property  ## https://zoujiu.blog.csdn.net/article/details/132531138
    def target_policies(self): 
        return [agt.target_actor for agt in self.agents]     ## 返回所有的目标策略网络

    def take_action(self, states, explore): ## 返回所有智能体采取的动作
        states = [                             ## 每个智能体的状态，放到一起的呢
            torch.tensor([states[i]], dtype=torch.float, device=self.device)
            for i in range(len(env.agents))
        ]
        return [
            agent.take_action(state, explore)                     ## 每个智能体根据状态返回动作的
            for agent, state in zip(self.agents, states)
        ]

    def update(self, sample, i_agent):
        '''
        用所有的智能体的下一个状态、策略网络输出下一个动作，合并到输入
        然后当前智能体的价值网络 输入所有内容，来给出价值判断，并 update 价值网络的
        所以每个智能体的价值网络update，需要所有智能体的 状态+动作，下一个状态+策略网络+下一个动作

        但是每个智能体的策略网络，只需要自身的 状态 + 策略网络 + 价值网络
        然后就可以update 自身的策略网络的

        也就是 中心化训练去中心化执行
        obs: [bs, 10-2], [bs, 10], [bs, 10]
        act: [bs,    5], [bs,  5], [bs,  5]
        rew: [bs,    1], [bs,  1], [bs,  1]
        '''
        obs, act, rew, next_obs, done = sample  ## 从采样内拿出 状态、动作、奖励、下一个状态、是否完成的
        cur_agent = self.agents[i_agent]        ## 拿到标号是第 i_agent 的智能体的

        cur_agent.critic_optimizer.zero_grad()   ## 当前 智能体 价值网络参数 的梯度 置0
        ## 拿到所有智能体的 策略网络输出的下一个状态的动作  [bs,    5], [bs,  5], [bs,  5]
        all_target_act = [    ## 下一个状态 输入到 目标策略网络，输出epsilon-贪婪算法来选择用哪个动作 
            onehot_from_logits(pi(_next_obs))    ## 目标策略网络
            for pi, _next_obs in zip(self.target_policies, next_obs)
        ]
        ## [bs, 2*20 + 2 + 1]
        target_critic_input = torch.cat((*next_obs, *all_target_act), dim=1) ## 所有智能体的下一个状态 + 所有智能体的下一个状态的动作
        ## cur_agent.target_critic() 所有智能体的下一个状态 + 所有智能体的下一个状态的动作，目标状态动作价值网络 输出（状态+动作）的动作价值 Q
        ## 用所有智能体的下一个（状态+动作）对的动作价值 Q，然后间接求出所有智能体当前的（状态+动作）对的动作价值
        target_critic_value = rew[i_agent].view(    ## [bs,  1]
            -1, 1) + self.gamma * cur_agent.target_critic(
                target_critic_input) * (1 - done[i_agent].view(-1, 1))
        ## [bs, 2*20 + 2 + 1]
        critic_input = torch.cat((*obs, *act), dim=1)      ## 所有智能体的: 当前的状态 + 当前的动作，价值网络的输入
        critic_value = cur_agent.critic(critic_input)      ## 所有智能体的: 直接求出当前（状态+动作）的动作价值 ## [bs,  1]
        ## 直接求出所有智能体当前（状态、动作）的动作价值，和 间接求出的所有智能体的动作价值，使用 MSE 来算损失函数的，target_critic_value不反向传播求梯度
        ## 用来 update 当前智能体的价值网络，输入是所有智能体
        critic_loss = self.critic_criterion(critic_value,
                                            target_critic_value.detach())
        critic_loss.backward()                   ## 价值网络的损失loss反向传播梯度
        cur_agent.critic_optimizer.step()        ## update 网络

        cur_agent.actor_optimizer.zero_grad()               ## 策略网络参数梯度置 0 的
        cur_actor_out = cur_agent.actor(obs[i_agent])       ## 当前智能体策略网络根据当前状态，给出动作的概率
        cur_act_vf_in = gumbel_softmax(cur_actor_out)       ## 动作采样的
        all_actor_acs = []                                    
        for i, (pi, _obs) in enumerate(zip(self.policies, obs)):      ## 遍历所有的策略 和 状态
            if i == i_agent:
                all_actor_acs.append(cur_act_vf_in)                   ## 是当前智能体，加入当前智能体的动作
            else:
                all_actor_acs.append(onehot_from_logits(pi(_obs)))    ## 其他智能体，策略网络根据当前状态，给出动作的，epsilon-贪婪算法来选择用哪个动作 
        ## [bs, 2*20 + 2 + 1]
        vf_in = torch.cat((*obs, *all_actor_acs), dim=1)              ## 拼接所有的当前状态 + 所有的动作的
        actor_loss = -cur_agent.critic(vf_in).mean()                  ## 最大化价值网络的输出，也就是最大化动作价值网络
        actor_loss += (cur_actor_out**2).mean() * 1e-3  ## 当前智能体策略网络根据当前状态，给出动作的概率，动作概率越小越好，探索性变强了的，正则化
        actor_loss.backward()       ## 策略网络反向传播梯度
        cur_agent.actor_optimizer.step()     ## 策略网络参数update

    def update_all_targets(self):         ## 延迟少量的update网络，也就是EMA的方式
        for agt in self.agents:
            agt.soft_update(agt.actor, agt.target_actor, self.tau)
            agt.soft_update(agt.critic, agt.target_critic, self.tau)

num_episodes = 6000 - 1000
episode_length = 60  # 每条序列的最大长度
buffer_size = 100000
hidden_dim = 100
actor_lr = 0.01
critic_lr = 0.01
gamma = 0.95
tau = 1e-2
batch_size = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
update_interval = 100
minimal_size = 4000

env_id = "simple_adversary"
env = make_env(env_id)  ## 产生多智能体的环境
env.reset()
replay_buffer = ReplayBuffer(buffer_size)  ## 历史数据的回放池

state_dims = []
action_dims = []
for action_space in env.action_space:       ## 遍历所有智能体的动作空间，拿到每一个智能体动作的 dim
    action_dims.append(action_space.n)
for state_space in env.observation_space:   ## 遍历所有智能体的状态空间，拿到每一个智能体状态的 dim
    state_dims.append(state_space.shape[0])
critic_input_dim = sum(state_dims) + sum(action_dims)   ## 两者 dim 的累加，也就是中心化Critic网络的输入的 dim

maddpg = MADDPG(env, device, actor_lr, critic_lr, hidden_dim, state_dims,    ## 实例化多智能体 DDPG algorithm
                action_dims, critic_input_dim, gamma, tau)

def evaluate(env_id, maddpg, n_episode=10, episode_length=60):
    # 对学习的策略进行评估,此时不会进行探索
    env = make_env(env_id)
    returns = np.zeros(len(env.agents))
    for _ in range(n_episode):
        obs = env.reset()
        for t_i in range(episode_length):
            actions = maddpg.take_action(obs, explore=False)
            obs, rew, done, info = env.step(actions)
            rew = np.array(rew)
            returns += rew / n_episode
    return returns.tolist()

return_list = []  # 记录每一轮的回报（return）
total_step = 0
allimage = []
for i_episode in range(num_episodes):
    state = env.reset()
    # ep_returns = np.zeros(len(env.agents))
    for e_i in range(episode_length):
        if i_episode > num_episodes - 30:
            img = env.render(mode = r'rgb_array')[0]
            if len(img.shape)!=3:
                continue
            allimage.append(img)
        actions = maddpg.take_action(state, explore=True)
        next_state, reward, done, _ = env.step(actions)
        replay_buffer.add(state, actions, reward, next_state, done)
        state = next_state

        total_step += 1
        if replay_buffer.size(
        ) >= minimal_size and total_step % update_interval == 0:
            sample = replay_buffer.sample(batch_size)

            def stack_array(x):
                rearranged = [[sub_x[i] for sub_x in x] for i in range(len(x[0]))]
                return [
                    torch.FloatTensor(np.vstack(aa)).to(device)
                    for aa in rearranged
                ]

            sample = [stack_array(x) for x in sample]
            for a_i in range(len(env.agents)):
                maddpg.update(sample, a_i)
            maddpg.update_all_targets()
    if (i_episode + 1) % 100 == 0:
        ep_returns = evaluate(env_id, maddpg, n_episode=100)
        return_list.append(ep_returns)
        print(f"Episode: {i_episode+1}, {ep_returns}")
        
# https://github.com/guicalare/Img2gif/blob/master/Code/Img2Gif.py
pth = r'C:\Users\10696\Desktop\access\Hands-on-RL'
imageio.mimsave(os.path.join(pth, 'chapter%s__.gif'%str(21)), allimage, duration=60)

return_array = np.array(return_list)
for i, agent_name in enumerate(["adversary_0", "agent_0", "agent_1"]):
    plt.figure()
    plt.plot(
        np.arange(return_array.shape[0]) * 100,
        rl_utils.moving_average(return_array[:, i], 9))
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title(f"{agent_name} by MADDPG")