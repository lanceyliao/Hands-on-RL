import random
import time

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


class DynaQ:
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_planning,
                 n_action=4):
        # n_action表示动作的个数，这里是4个动作
        self.Q_table = np.zeros([nrow * ncol, n_action])  # Q表
        self.n_action = n_action
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.n_planning = n_planning
        self.model = {}  # 用于存储模型，格式为

    # 根据epsilon-greedy策略选择动作
    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def q_learning_update(self, s0, a0, r, s1):
        td_error = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error

    # # 根据当前状态的Q表，选择最优动作
    # def best_action(self, state):
    #     Q_max = np.max(self.Q_table[state])
    #     state_action_array = [0 for _ in range(self.n_action)]
    #     for i in range(self.n_action):
    #         if self.Q_table[state, i] == Q_max:
    #             state_action_array[i] = 1
    #     return state_action_array
    def update(self, s0, a0, r, s1):
        self.q_learning_update(s0, a0, r, s1)
        self.model[(s0, a0)] = (r, s1)
        for _ in range(self.n_planning):
            (s, a), (r, s_) = random.choice(list(self.model.items()))
            self.q_learning_update(s, a, r, s_)


def DynaQ_CliffWalking(n_planning):
    ncol = 12
    nrow = 4
    env = CliffWalkingEnv(ncol, nrow)

    epsilon = 0.01  # 贪婪度
    alpha = 0.1  # 学习率
    gamma = 0.9  # 折扣因子
    agent = DynaQ(ncol, nrow, epsilon, alpha, gamma, n_planning)
    num_episodes = 300

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i:d}') as pbar:
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
    return return_list


np.random.seed(0)
random.seed(0)
n_planning_list = [0, 2, 20]
for n_planning in n_planning_list:
    print(f'Q-planning步数为{n_planning:d}')
    time.sleep(0.5)
    return_list = DynaQ_CliffWalking(n_planning)
    episode_list = list(range(len(return_list)))
    plt.plot(episode_list, return_list, label=f'{n_planning} planning steps')
plt.xlabel('Episode')
plt.ylabel('Returns')
plt.legend()  # 显示图例
plt.title('Dyna-Q on Cliff Walking')
plt.show()
