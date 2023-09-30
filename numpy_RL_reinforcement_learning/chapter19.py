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
import collections
import numpy as np
import matplotlib.pyplot as plt
import rl_utils
from tqdm import tqdm
from net.fullconnect import fclayer
from net.activation import Softmax, ReLU, tanh, sigmoid
from net.loss import mean_square_two_derive_loss, binary_cross_entropy_loss
import os
import imageio
import cv2
import torch
import torch.nn.functional as F

distance_threshold = 0.06
class WorldEnv:
    def __init__(self):
        global distance_threshold
        self.distance_threshold = distance_threshold  ## 距离的最小阀值
        self.action_bound = 1  ## 动作的上界的

    def reset(self):  # 重置环境
        # 生成一个目标状态, 坐标范围是[3.5～4.5, 3.5～4.5]
        self.goal = np.array(
            [4 + random.uniform(-0.5, 0.5), 4 + random.uniform(-0.5, 0.5)])  ## 随机给定目标g
        self.state = np.array([0, 0])  # 初始状态
        self.count = 0 ## 重置步长的
        return np.hstack((self.state, self.goal))  ## 叠起来状态和目标

    def step(self, action):
        action = np.squeeze(np.clip(action, -self.action_bound, self.action_bound))  ## 截断动作的上下界
        x = max(0, min(5, self.state[0] + action[0])) ## 前进一步以后，截断x坐标
        y = max(0, min(5, self.state[1] + action[1])) ## 前进一步以后，截断y坐标
        self.state = np.array([x, y])                 ## ndarray 化
        self.count += 1  ## 累积步长的呢

        dis = np.sqrt(np.sum(np.square(self.state - self.goal)))  ## 算当前状态和目标 g 的距离
        reward = -1.0 if dis > self.distance_threshold else 0  ## > 给定阀值奖励是 -1，否则 0
        if dis <= self.distance_threshold or self.count == 50: ## 若 < 阀值，或者步长一定，就完成
            done = True ## 完成
        else:
            done = False  ## 没有完成的

        return np.hstack((self.state, self.goal)), reward, done ## 叠起来状态和目标，奖励，是否完成的

##  构造策略网络的
class PolicyNet():
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim, adam = True):
        super(PolicyNet, self).__init__()
        self.fc1 = fclayer(state_dim, hidden_dim, adam=adam)
        self.fc2 = fclayer(hidden_dim, hidden_dim, adam=adam)
        self.fc3 = fclayer(hidden_dim, action_dim, adam=adam)
        self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值
        self.tanh = tanh()
        # self.tanh = torch.nn.Tanh().requires_grad_(True)
        self.relu0 = ReLU()
        self.relu1 = ReLU()

    def forward(self, x):
        self.x = x
        self.x1 = self.fc1.forward(x)
        self.x2 = self.relu0.forward(self.x1)  # 隐藏层使用ReLU激活函数
        self.x3 = self.fc2.forward(self.x2)
        self.x9 = self.relu1.forward(self.x3)
        self.x3 = self.fc3.forward(self.x9)
        self.x6 = self.tanh.forward(self.x3)
        # self.k = torch.tensor(self.x3).requires_grad_(True)
        # self.x6 = self.tanh(self.k).requires_grad_(True)
        # self.x00 = self.x6.detach().cpu().numpy() * self.action_bound
        self.x00 = self.x6 * self.action_bound
        return self.x00
    
    def backward(self, delta):
        delta = delta * self.action_bound
        delta = self.tanh.backward(delta)
        delta = self.fc3.backward(delta, inputs=self.x9)
        # kk = torch.sum(self.x6).requires_grad_(True)
        # dd = torch.autograd.grad(kk, self.k, create_graph=True)[0].detach().cpu().numpy()
        # delta = delta * dd
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
    
    def setparam(self, model):
        self.fc1.params = model.fc1.params.copy()
        self.fc2.params = model.fc2.params.copy()
        self.fc3.params = model.fc3.params.copy()
        if self.fc1.bias:
            self.fc1.bias_params = model.fc1.bias_params.copy()
        if self.fc2.bias:
            self.fc2.bias_params = model.fc2.bias_params.copy()
        if self.fc3.bias:
            self.fc3.bias_params = model.fc3.bias_params.copy()

## 状态动作价值网络，输出状态动作对 (状态，动作) 的价值
## 也就是动作价值网络
class QValueNet():
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim, adam = True):
        super(QValueNet, self).__init__()
        self.fc1 = fclayer(state_dim + action_dim, hidden_dim, adam=adam)
        self.fc2 = fclayer(hidden_dim, hidden_dim, adam=adam)
        self.fc_out = fclayer(hidden_dim, 1, adam=adam)
        self.relu0 = ReLU()
        self.relu1 = ReLU()

    ## 输入 (状态，动作)
    def forward(self, x, a):
        cat = np.concatenate([x, a], axis=1) # 拼接状态和动作
        self.x = cat
        self.x1 = self.fc1.forward(self.x)
        self.x2 = self.relu0.forward(self.x1)  # 隐藏层使用ReLU激活函数
        self.x3 = self.fc2.forward(self.x2)
        self.x6 = self.relu1.forward(self.x3)
        self.x00 = self.fc_out.forward(self.x6)
        return self.x00
    
    def backward(self, delta):
        delta = self.fc_out.backward(delta, inputs=self.x6)
        delta = self.relu1.backward(delta)
        delta = self.fc2.backward(delta, inputs=self.x2)
        delta = self.relu0.backward(delta)
        delta = self.fc1.backward(delta, inputs=self.x)
        return delta
    
    def update(self, lr):
        self.fc_out.update(lr)
        self.fc2.update(lr)
        self.fc1.update(lr)
    
    def setzero(self):
        self.fc_out.setzero()
        self.fc2.setzero()
        self.fc1.setzero()

    def save_model(self):
        return [self.fc1.save_model(), self.fc2.save_model(), self.fc_out.save_model()]
        
    def restore_model(self, models):
        self.fc1.restore_model(models[0])
        self.fc2.restore_model(models[1])
        self.fc_out.restore_model(models[2])
    
    def setparam(self, model):
        self.fc1.params = model.fc1.params.copy()
        self.fc2.params = model.fc2.params.copy()
        self.fc_out.params = model.fc_out.params.copy()
        if self.fc1.bias:
            self.fc1.bias_params = model.fc1.bias_params.copy()
        if self.fc2.bias:
            self.fc2.bias_params = model.fc2.bias_params.copy()
        if self.fc_out.bias:
            self.fc_out.bias_params = model.fc_out.bias_params.copy()

class DDPG:
    ''' DDPG算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, actor_lr, critic_lr, sigma, tau, gamma):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound) ##  策略网络的
        self.critic = QValueNet(state_dim, hidden_dim, action_dim) ## 状态动作价值网络
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound) ##  目标策略网络，延迟update
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim) ##  目标状态动作价值网络，延迟update
        # 初始化目标价值网络并设置和价值网络相同的参数
        self.target_critic.setparam(self.critic)
        # 初始化目标策略网络并设置和策略相同的参数
        self.target_actor.setparam(self.actor)
        self.gamma = gamma  ## 折扣因子
        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.tau = tau  # 目标网络软更新参数
        self.action_dim = action_dim  ## 动作的dim
        self.actor_lr  = actor_lr
        self.critic_lr = critic_lr
    
    def take_action(self, state):
        state = np.array([state], dtype=np.float32)
        action = self.actor.forward(state)[0]   ## 拿到确定性策略 网络的输出，也就是确定的动作
        # 给动作添加噪声，增加探索
        action = action + self.sigma * np.random.randn(self.action_dim)
        return action

    def soft_update_Value(self, net, target_net):   ## EMA的方式update目标网络的，每次update很少的内容
        target_net.fc1.params = (target_net.fc1.params * (1.0 - self.tau) + net.fc1.params * self.tau)
        target_net.fc2.params = (target_net.fc2.params * (1.0 - self.tau) + net.fc2.params * self.tau)
        target_net.fc_out.params = (target_net.fc_out.params * (1.0 - self.tau) + net.fc_out.params * self.tau)
        
        target_net.fc1.bias_params = (target_net.fc1.bias_params * (1.0 - self.tau) + net.fc1.bias_params * self.tau)
        target_net.fc2.bias_params = (target_net.fc2.bias_params * (1.0 - self.tau) + net.fc2.bias_params * self.tau)
        target_net.fc_out.bias_params = (target_net.fc_out.bias_params * (1.0 - self.tau) + net.fc_out.bias_params * self.tau)

    def soft_update_Policy(self, net, target_net):   ## EMA的方式update目标网络的，每次update很少的内容
        target_net.fc1.params = (target_net.fc1.params * (1.0 - self.tau) + net.fc1.params * self.tau)
        target_net.fc2.params = (target_net.fc2.params * (1.0 - self.tau) + net.fc2.params * self.tau)
        target_net.fc3.params = (target_net.fc3.params * (1.0 - self.tau) + net.fc3.params * self.tau)
        
        target_net.fc1.bias_params = (target_net.fc1.bias_params * (1.0 - self.tau) + net.fc1.bias_params * self.tau)
        target_net.fc2.bias_params = (target_net.fc2.bias_params * (1.0 - self.tau) + net.fc2.bias_params * self.tau)
        target_net.fc3.bias_params = (target_net.fc3.bias_params * (1.0 - self.tau) + net.fc3.bias_params * self.tau)

    def update(self, transition_dict):
        ## 拿到这条序列内的 状态和目标、动作和奖励，下一个状态和目标、是否完成的
        states = np.array(transition_dict['states'])
        actions = np.array(transition_dict['actions'])
        rewards = np.array(transition_dict['rewards']).reshape((-1, 1))
        next_states = np.array(transition_dict['next_states'])
        dones = np.array(transition_dict['dones']).reshape((-1, 1))

        ## 下个状态和目标+下个状态目标的动作，目标状态和目标动作价值网络 输出（状态和目标，动作）对的动作价值 Q 
        next_q_values = self.target_critic.forward(next_states, self.target_actor.forward(next_states))
        ## 用下一个（状态和目标、动作）对的动作价值 Q，然后间接求出当前的（状态和目标、动作）对的动作价值
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        ## 直接求出当前（状态和目标的、动作）的动作价值，和 间接求出的动作价值，使用 MSE 来算损失函数的，q_targets不反向传播求梯度
        nowval = self.critic.forward(states, actions)
        critic_loss, nowval_delta, _ = mean_square_two_derive_loss(nowval, q_targets)    ## td_target.detach，不需要求梯度的
        self.critic.setzero()  ## 默认梯度会累积,这里需要显式将梯度置为0
        self.critic.backward(nowval_delta)  ##  反向传播求出 critic 的梯度
        self.critic.update(self.critic_lr) ##  使用累加的梯度来update参数

        ## update以后的动作价值网络，使用当前的（状态和目标的、动作）的动作价值，加了负号的，也就是最大化动作价值的
        ## self.actor(states) 是输出当前状态的动作
        '''
         确定性策略网络的 update，稍稍复杂些，嵌套了两个网络，首先用当前的状态和策略网络，输出当前状态的动作predict_A=self.actor(states)，
         然后 self.critic(states, self.actor(states))输出了当前（状态和目标的、动作）的动作价值Q，并求均值加负号，也就是最大化动作价值，
         反向传播用来update 策略网络actor，因critic动作价值网络上面已经update过了，所以下面就只update策略网络actor，critic网络不变的。
        '''
        actret = self.critic.forward(states, self.actor.forward(states))
        num = 1
        for i in actret.shape:
            num = num * i
        actor_loss = -np.mean(actret)
        delta = np.ones_like(actret) * (-1) / num
        self.critic.setzero()
        delta = self.critic.backward(delta)
        delta = delta[:, -self.action_dim:]   # np.expand_dims(delta[:, -2:], -1)

        self.actor.setzero()  ## 默认梯度会累积,这里需要显式将梯度置为0
        self.actor.backward(delta)  ##  反向传播求出 critic 的梯度
        self.actor.update(self.actor_lr) ##  使用累加的梯度来update参数
        ## 延迟少量的update网络，也就是EMA的方式
        self.soft_update_Policy(self.actor, self.target_actor)  # 软更新策略网络
        self.soft_update_Value(self.critic, self.target_critic)  # 软更新价值网络

class Trajectory:
    ''' 用来记录一条完整轨迹 '''
    def __init__(self, init_state):
        self.states = [init_state] ## 初始状态的
        self.actions = []
        self.rewards = []
        self.dones = []
        self.length = 0 ## 序列的长度

    def store_step(self, action, state, reward, done):
        self.actions.append(action)  ## 加入动作的呢
        self.states.append(state)
        self.rewards.append(reward)
        self.dones.append(done)
        self.length += 1

class ReplayBuffer_Trajectory:
    ''' 存储轨迹的经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  ## 初始化缓冲池

    def add_trajectory(self, trajectory):
        self.buffer.append(trajectory)    ## 缓冲池加入一条序列

    def size(self):
        return len(self.buffer)   ## 缓冲池的长度

    def sample(self, batch_size, use_her, her_ratio=0.8):
        global distance_threshold
        batch = dict(states=[],               ## 批量
                     actions=[],
                     next_states=[],
                     rewards=[],
                     dones=[])
        for _ in range(batch_size):
            traj = random.sample(self.buffer, 1)[0]      ## 采样一条序列
            step_state = np.random.randint(traj.length)  ## 随机拿序列的某一个record，坐标是 step_state
            state = traj.states[step_state]              ## 该record的状态和目标
            next_state = traj.states[step_state + 1]     ## 下一个状态和目标的
            action = traj.actions[step_state]            ## 该record的动作
            reward = traj.rewards[step_state]            ## 该record的奖励
            done = traj.dones[step_state]                ## 该record是否完成的
            ## use_her是否使用方式 her，概率值小于阀值
            if use_her and np.random.uniform() <= her_ratio:   ## 给定好 使用 her 方式的概率
                step_goal = np.random.randint(step_state + 1, traj.length + 1)  ## 从在同一个轨迹并在时间上处在s'之后的某个状态s''，坐标是step_goal
                ## 也就是目标的(x,y)坐标 g'
                goal = traj.states[step_goal][:2]  # 使用HER算法的future方案设置目标 g'，也就是后面的某个状态做目标
                ## 配置修改以后的
                dis = np.sqrt(np.sum(np.square(next_state[:2] - goal)))  ## 下一个状态的目标 g_next(x,y)，和配置的目标 g'之间的距离
                reward = -1.0 if dis > distance_threshold else 0  ## 大于阀值则奖励 -1，小于阀值则奖励 0
                done = False if dis > distance_threshold else True  ## 大于阀值则未完成，小于阀值则已完成
                '''
                像状态state和next_state，前两个数字是状态，后两个数字是目标
                两者的状态改变值很小，所以train起来更加方便，模型只需要一步一步接近目标就可以
                也就是模型只需要学着一步一步靠近目标，
                而不需要一次跨越大的步长
                来到达目的地的呢。
                '''
                state = np.hstack((state[:2], goal))  ## 状态和配置的目标 g' 组合到一起的
                ## 因state和next_state两者挨得很近，所以状态也很近，模型训练起来更加方便的此时的目标变得更近了
                next_state = np.hstack((next_state[:2], goal))  ## 下一个状态和配置的目标g' 组合到一起的，最后可以用来做下一个状态
            ## 放入到采样的批量内
            batch['states'].append(state)
            batch['next_states'].append(next_state)
            batch['actions'].append(action)
            batch['rewards'].append(reward)
            batch['dones'].append(done)
        ## ndarray化
        batch['states'] = np.array(batch['states'])
        batch['next_states'] = np.array(batch['next_states'])
        batch['actions'] = np.array(batch['actions'])
        return batch

actor_lr = 1e-3
critic_lr = 1e-3
hidden_dim = 128
state_dim = 4
action_dim = 2
action_bound = 1
sigma = 0.1
tau = 0.005
gamma = 0.98
num_episodes = 2000
n_train = 20
batch_size = 256
minimal_episodes = 200
buffer_size = 10000

random.seed(6)
np.random.seed(6)
env = WorldEnv()
replay_buffer = ReplayBuffer_Trajectory(buffer_size)
agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, actor_lr,
             critic_lr, sigma, tau, gamma)

return_list = []
allimage = []
epoch = 10
imgsize = 600
for i in range(epoch):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0   ## 累加的奖励值
            state = env.reset()  ## 重置环境的呢
            if i == epoch - 1:
                tta = [int(k * imgsize / (6-1)) for k in state[2:]]
            traj = Trajectory(state)  ## 实例化''' 用来记录一条完整轨迹 '''
            done = False         ## 是否完成的呢
            cnt = 0
            while not done:
                if i == epoch - 1:
                    img = np.ones((imgsize, imgsize, 3), dtype = np.uint8) * (2**8 - 1)
                    ttt = [int(k * imgsize / (6-1)) for k in state[:2]]
                    # cv2.imshow('i', img)
                    # cv2.waitKey(-1)
                    cv2.circle(img, (ttt[0], ttt[1]), 9, (0, 0, 2**8-1), 2)
                    cv2.circle(img, (tta[0], tta[1]), 6 + int(imgsize * distance_threshold), (2**8-1, 0, 0), 1)
                    cv2.circle(img, (tta[0], tta[1]), 6, (2**8-1, 0, 0), 1)
                    cv2.putText(img, text=r"D%d"%i_episode, org=(tta[0], tta[1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1, color=(2**8-1, 160, 160), thickness=2)
                    # cv2.imshow('i', img)
                    # cv2.waitKey(-1)
                    allimage.append(img)
                action = agent.take_action(state)   ## 智能体根据状态选择 动作
                state, reward, done = env.step(action)  ## 环境执行动作，并返回下一步的状态和目标、奖励、是否完成的
                episode_return += reward  ## 累积奖励的
                traj.store_step(action, state, reward, done)  ## 保存轨迹到序列内，包括（动作、状态和目标、奖励、是否完成的）
            replay_buffer.add_trajectory(traj)  ## 回放池加入这条 episode
            return_list.append(episode_return)
            if replay_buffer.size() >= minimal_episodes:  ## 当回放池内的个数 > 最小值
                for _ in range(n_train):
                    transition_dict = replay_buffer.sample(batch_size, True)   ## 按照常规采样 和 HER 来进行采样的
                    agent.update(transition_dict)  ## 智能体使用采样的序列来 train 策略网络和价值网络
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)
    if i == epoch - 1:
        # https://github.com/guicalare/Img2gif/blob/master/Code/Img2Gif.py
        pth = r'C:\Users\10696\Desktop\access\Hands-on-RL\numpy_RL_reinforcement_learning'
        imageio.mimsave(os.path.join(pth, 'chapter%s.gif'%str("19_HER")), allimage, duration=190)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DDPG with HER on {}'.format('GridWorld'))
plt.show()

random.seed(160)
np.random.seed(160)
distance_threshold = 0.16
env = WorldEnv()
replay_buffer = ReplayBuffer_Trajectory(buffer_size)
agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, actor_lr,
             critic_lr, sigma, tau, gamma)

return_list = []
allimage = []
epoch = 10
imgsize=600
for i in range(epoch):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            state = env.reset()
            if i == epoch - 1 and i_episode > int(num_episodes / 10)-60:
                tta = [int(k * imgsize / (6-1)) for k in state[2:]]
            traj = Trajectory(state)
            done = False
            while not done:
                if i == epoch - 1 and i_episode > int(num_episodes / 10)-60:
                    img = np.ones((imgsize, imgsize, 3), dtype = np.uint8) * (2**8 - 1)
                    ttt = [int(k * imgsize / (6-1)) for k in state[:2]]
                    # cv2.imshow('i', img)
                    # cv2.waitKey(-1)
                    cv2.circle(img, (ttt[0], ttt[1]), 9, (0, 0, 2**8-1), 2)
                    cv2.circle(img, (tta[0], tta[1]), 6 + int(imgsize * distance_threshold), (2**8-1, 0, 0), 1)
                    cv2.circle(img, (tta[0], tta[1]), 6, (2**8-1, 0, 0), 1)
                    cv2.putText(img, text=r"D%d"%i_episode, org=(tta[0], tta[1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1, color=(2**8-1, 160, 160), thickness=2)
                    # cv2.imshow('i', img)
                    # cv2.waitKey(-1)
                    allimage.append(img)
                action = agent.take_action(state)
                state, reward, done = env.step(action)
                episode_return += reward
                traj.store_step(action, state, reward, done)
            replay_buffer.add_trajectory(traj)
            return_list.append(episode_return)
            if replay_buffer.size() >= minimal_episodes:
                for _ in range(n_train):
                    # 和使用HER训练的唯一区别
                    transition_dict = replay_buffer.sample(batch_size, False) ## 不使用 HER 采样的
                    agent.update(transition_dict)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)
    if i == epoch - 1:
        # https://github.com/guicalare/Img2gif/blob/master/Code/Img2Gif.py
        pth = r'C:\Users\10696\Desktop\access\Hands-on-RL\numpy_RL_reinforcement_learning'
        imageio.mimsave(os.path.join(pth, 'chapter%s.gif'%str("19NO_HER")), allimage, duration=30)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DDPG without HER on {}'.format('GridWorld'))
plt.show()