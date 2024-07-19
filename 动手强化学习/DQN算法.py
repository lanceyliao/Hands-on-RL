import random, gym, numpy as np,collections, torch,torch.nn.functional as F, matplotlib.pyplot as plt,rl_utils
from tqdm import tqdm

class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))  # 队列中的最后一个元素是最新的一个数据

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)  # 从buffer中随机采样batch_size个数据
        state, action, reward, next_state, done = zip(*transitions)  # 将采样的数据分别解压为state, action, reward, next_state, done
        return np.array(state), action, reward, np.array(next_state), done  # 返回这些数据

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)


class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        # 调用父类的构造器
        super(Qnet, self).__init__()
        # 创建两个全连接层，一个输入层，一个输出层
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # 隐藏层使用ReLU激活函数
        x = F.relu(self.fc1(x))
        # 输出层不使用激活函数，直接输出
        return self.fc2(x)

class DQN:
    ''' DQN算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim,
                          self.action_dim).to(device)  # Q网络
        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
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

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
            -1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones
                                                                )  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1

lr = 2e-3 # 学习率
num_episodes = 500 # 训练回合数
hidden_dim = 128 # 隐藏层节点数
gamma = 0.98 # 折扣因子
epsilon = 0.01 # 探索因子
target_update = 10 # target网络更新的回合间隔
buffer_size = 10000 # 经验回放池大小
minimal_size = 500 # 经验回放池最小大小
batch_size = 64 # 批大小
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") # GPU加速

env_name = 'CartPole-v0' # 环境名称
env = gym.make(env_name) # 创建环境
random.seed(0) # 设置随机种子
np.random.seed(0)
env.seed(0)
torch.manual_seed(0)
replay_buffer = ReplayBuffer(buffer_size) # 创建经验回放池
state_dim = env.observation_space.shape[0] # 状态维度
action_dim = env.action_space.n # 动作维度
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device) # 创建DQN网络

return_list = [] # 记录每个回合的累计回报
for i in range(10): # 迭代10次
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar: # 进度条
        for i_episode in range(int(num_episodes / 10)): # 每次迭代执行10个episode
            episode_return = 0 # 每个episode的回报
            state = env.reset() # 重置环境
            done = False # 是否episode结束
            while not done: # 当episode未结束时
                action = agent.take_action(state) # 选择动作
                next_state, reward, done, _ = env.step(action) # 执行动作
                replay_buffer.add(state, action, reward, next_state, done) # 将transition加入buffer
                state = next_state # 更新状态
                episode_return += reward # 更新回报
                # 当buffer数据的数量超过一定值后,才进行Q网络训练
                if replay_buffer.size() > minimal_size: # buffer数据量超过最小值
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size) # 从buffer中采样数据
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    agent.update(transition_dict) # Q网络训练
            return_list.append(episode_return) # 记录每个episode的回报
            if (i_episode + 1) % 10 == 0: # 每训练10个episode,打印一次进度
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()

class ConvolutionalQnet(torch.nn.Module):
    ''' 加入卷积层的Q网络 '''
    def __init__(self, action_dim, in_channels=4):
        super(ConvolutionalQnet, self).__init__()
        # 初始化卷积层
        self.conv1 = torch.nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # 初始化全连接层
        self.fc4 = torch.nn.Linear(7 * 7 * 64, 512)
        # 初始化输出层
        self.head = torch.nn.Linear(512, action_dim)

    def forward(self, x):
        # 将输入归一化到[0,1]
        x = x / 255
        # 激活函数ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x))
        # 输出层不需要激活函数
        return self.head(x)