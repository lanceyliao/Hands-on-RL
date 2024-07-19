import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class CliffWalkingEnv:
    def __init__(self, ncol, nrow):
        # 初始化环境，定义行数和列数
        self.nrow = nrow
        self.ncol = ncol
        # 初始化位置为左下角
        self.x = 0
        self.y = self.nrow - 1

    def reset(self):
        # 重置位置为左下角
        self.x = 0
        self.y = self.nrow - 1
        # 返回位置的序号
        return self.y * self.ncol + self.x

    def step(self, action):
        # 定义上下左右的移动
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        # 更新位置
        self.x = max(0, min(self.ncol - 1, self.x + change[action][0]))
        self.y = max(0, min(self.nrow - 1, self.y + change[action][1]))
        # 返回新的位置序号，奖励，是否结束
        next_state = self.y * self.ncol + self.x
        reward = -1
        done = False
        # 判断是否到达悬崖
        if self.y == self.nrow - 1 and self.x > 0:
            done = True
            # 判断是否到达目标位置
            if self.x != self.ncol - 1:
                reward = -100
        return next_state, reward, done


class Sarsa:
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        # n_action表示动作的个数，这里是4个动作
        self.Q_table = np.zeros([nrow * ncol, n_action])  # Q表
        self.n_action = n_action
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def take_action(self, state):  # 根据epsilon-greedy策略选择动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state):  # 根据当前状态的Q表，选择最优动作
        Q_max = np.max(self.Q_table[state])
        state_action_array = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                state_action_array[i] = 1
        return state_action_array

    def update(self, s0, a0, r, s1, a1):  # 根据Sarsa算法来更新Q表
        td_error = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error


# 1. 初始化环境和agent
ncol = 12
nrow = 4
env = CliffWalkingEnv(ncol, nrow)
np.random.seed(0)
epsilon = 0.1
alpha = 0.1
gamma = 0.9
agent = Sarsa(ncol, nrow, epsilon, alpha, gamma)
num_episodes = 500

return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i:d}') as pbar:
        for i_episode in range(int(num_episodes / 10)):
            # 2. 重置环境并选择初始动作
            episode_return = 0
            state = env.reset()  # 重置环境
            action = agent.take_action(state)  # 选择初始动作
            done = False
            while not done:
                # 3. 使用env的位置更新函数，传入动作，返回下一状态和奖励
                next_state, reward, done = env.step(action)  # 执行动作，返回下一状态，奖励和游戏是否结束
                # 4. 使用agent的epsilon-greedy动作生成，传入下一状态，返回下一动作
                next_action = agent.take_action(next_state)  # 选择下一动作
                # 5. 更新Q表
                agent.update(state, action, reward, next_state, next_action)  # 更新Q表
                # 6. 更新状态和动作
                state = next_state  # 更新状态
                action = next_action  # 更新动作
                # 7. 累加回报
                episode_return += reward  # 计算回报
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({'episode': f'{int(num_episodes / 10 * i + i_episode + 1):d}',
                                  'return': f'{np.mean(return_list[-10:]):.3f}'})
                pbar.update(1)
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Sarsa on Cliff Walking')
plt.show()


def print_agent(agent, env, action_meaning, disaster=[], end=[]):
    for i in range(env.nrow):
        for j in range(env.ncol):
            state = i * env.ncol + j
            # 如果当前状态是障碍物，用'****'表示
            if state in disaster:
                print('****', end=' ')
            # 如果当前状态是终止状态，用'EEEE'表示
            elif state in end:
                print('EEEE', end=' ')
            else:
                # 找到当前状态的最优动作
                state_action_array = agent.best_action(state)
                state_action_str = ''
                # 将每个动作对应的最优动作用action_meaning中的字符表示
                for k in range(len(state_action_array)):
                    state_action_str += action_meaning[k] if state_action_array[k] else ' '
                print(state_action_str, end=' ')
        print()


action_meaning = ['↑', '↓', '←', '→']
print('Sarsa算法最终收敛得到的策略：')
# 将障碍物的状态和终止状态传入，用于在打印时，将这些状态用特殊字符表示
print_agent(agent, env, action_meaning, disaster=list(range(37, 47)), end=[47])


# 多步Sarsa算法
class nstep_Sarsa:
    def __init__(self, n, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        # n_action表示动作的个数，这里是4个动作
        self.Q_table = np.zeros([nrow * ncol, n_action])  # Q表
        self.n_action = n_action
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # 相对于Sarsa增加的参数
        self.n = n
        self.state_list = []  # 用于存储状态
        self.action_list = []  # 用于存储动作
        self.reward_list = []  # 用于存储奖励

    def take_action(self, state):  # 根据epsilon-greedy策略选择动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state):  # 根据当前状态的Q表，选择最优动作
        Q_max = np.max(self.Q_table[state])
        state_action_array = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                state_action_array[i] = 1
        return state_action_array

    def update(self, s0, a0, r, s1, a1, done):  # done表示游戏是否结束
        # 将当前状态、动作、奖励存入列表
        self.state_list.append(s0)
        self.action_list.append(a0)
        self.reward_list.append(r)
        # 如果列表中的元素个数可以构成一个n步，就更新Q表
        if len(self.state_list) == self.n:
            G = self.Q_table[s1, a1]
            for i in reversed(range(self.n)):
                G = self.gamma * G + self.reward_list[i]
                if done and i > 0:
                    s = self.state_list[i]
                    a = self.action_list[i]
                    self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])
            s = self.state_list.pop(0)
            a = self.action_list.pop(0)
            self.reward_list.pop(0)
            self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])
        if done:
            self.state_list.clear()  # 清空列表
            self.action_list.clear()
            self.reward_list.clear()


np.random.seed(0)
n_step = 5
epsilon = 0.1
alpha = 0.1
gamma = 0.9
agent = nstep_Sarsa(n_step, ncol, nrow, epsilon, alpha, gamma)
num_episodes = 500

return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            # 2. 重置环境并选择初始动作
            episode_return = 0
            state = env.reset()  # 重置环境
            action = agent.take_action(state)  # 选择初始动作
            done = False
            while not done:
                # 3. 使用env的位置更新函数，传入动作，返回下一状态和奖励
                next_state, reward, done = env.step(action)  # 执行动作，返回下一状态，奖励和游戏是否结束
                # 4. 使用agent的epsilon-greedy动作生成，传入下一状态，返回下一动作
                next_action = agent.take_action(next_state)  # 选择下一动作
                # 7. 累加回报
                episode_return += reward  # 计算回报
                # 5. 更新Q表
                agent.update(state, action, reward, next_state, next_action, done)  # 更新Q表
                # 6. 更新状态和动作
                state = next_state  # 更新状态
                action = next_action  # 更新动作
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({'episode': f'{int(num_episodes / 10 * i + i_episode + 1):d}',
                                  'return': f'{np.mean(return_list[-10:]):.3f}'})
                pbar.update(1)
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Sarsa on Cliff Walking')
plt.show()

action_meaning = ['↑', '↓', '←', '→']
print('5步Sarsa算法的最优策略：')
# 将障碍物的状态和终止状态传入，用于在打印时，将这些状态用特殊字符表示
print_agent(agent, env, action_meaning, disaster=list(range(37, 47)), end=[47])


class Q_learning:  # Q-learning算法
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        # n_action表示动作的个数，这里是4个动作
        self.Q_table = np.zeros([nrow * ncol, n_action])  # Q表
        self.n_action = n_action
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    # 根据epsilon-greedy策略选择动作
    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action
    # 根据当前状态的Q表，选择最优动作
    def best_action(self, state):
        Q_max = np.max(self.Q_table[state])
        state_action_array = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                state_action_array[i] = 1
        return state_action_array
    # 根据Sarsa算法来更新Q表
    def update(self, s0, a0, r, s1):
        td_error = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error


np.random.seed(0)
epsilon = 0.1  # 贪婪度
alpha = 0.1  # 学习率
gamma = 0.9  # 折扣因子
agent = Q_learning(ncol, nrow, epsilon, alpha, gamma)
num_episodes = 500

return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            # 重置环境并选择初始动作
            episode_return = 0
            state = env.reset()  # 重置环境
            done = False
            while not done:
                action = agent.take_action(state)  # 选择下一动作
                next_state, reward, done = env.step(action)  # 执行动作，返回下一状态，奖励和游戏是否结束
                agent.update(state, action, reward, next_state)  # 更新Q表
                state = next_state  # 更新状态
                episode_return += reward  # 计算回报
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({'episode': f'{int(num_episodes / 10 * i + i_episode + 1):d}',
                                  'return': f'{np.mean(return_list[-10:]):.3f}'})
                pbar.update(1)
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Q-learning on Cliff Walking')
plt.show()

action_meaning = ['↑', '↓', '←', '→']
print('Q-learning算法最终收敛得到的策略：')
# 将障碍物的状态和终止状态传入，用于在打印时，将这些状态用特殊字符表示
print_agent(agent, env, action_meaning, disaster=list(range(37, 47)), end=[47])
