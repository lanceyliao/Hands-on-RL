import os
from tqdm import tqdm
import numpy as np
import torch
import collections
import random
import imageio

# class ReplayBuffer:
#     def __init__(self, capacity):
#         self.buffer = collections.deque(maxlen=capacity)
#
#     def add(self, state, action, reward, next_state, done):
#         self.buffer.append((state, action, reward, next_state, done))
#
#     def sample(self, batch_size):
#         transitions = random.sample(self.buffer, batch_size)
#         state, action, reward, next_state, done = zip(*transitions)
#         return np.array(state), action, reward, np.array(next_state), done
#
#     def size(self):
#         return len(self.buffer)


class ReplayBuffer:
    '''
    经验回放池：最近的capacity个历史数据的保存、采样
    '''

    def __init__(self, capacity):  ## 容量的大小
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, s, a, r, s_, done):  # 将数据加入buffer
        self.buffer.append((s, a, r, s_, done))  ## 加入到队列内部，队列中的最后一个元素是最新的一个数据

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        b_P = random.sample(self.buffer, batch_size)  ## 随机采样的呢，拿到采样的历史数据。
        s, a, r, s_, done = zip(*b_P)  ## 使用zip来转置，也就是不同的自变量在不同的行
        b_s, b_a, b_r, b_s_, b_d = np.array(s), a, r, np.array(s_), done  ## 状态序列、动作序列、奖励序列、下一个状态序列，是否结束的序列
        return {'b_s': b_s, 'b_a': b_a, 'b_r': b_r, 'b_s_': b_s_, 'b_d': b_d}

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)  ##  返回历史数据的总长度

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

# def train_on_policy_agent(env, agent, num_episodes):
#     return_list = []
#     for i in range(10):
#         with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
#             for i_episode in range(int(num_episodes/10)):
#                 episode_return = 0
#                 transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
#                 state = env.reset()
#                 if len(state)!=2*2:
#                     state = state[0]
#                 done = False
#                 ## 采样一条序列的
#                 while not done:
#                     action = agent.take_action(state)    ##  根据状态采取动作的
#                     ##  环境执行动作，并反馈下一个状态、动作的奖励、是否完成、步长太长的，info
#                     next_state, reward, terminated, truncated, info = env.step(action)
#                     done = terminated | truncated       ## 终止或者步长太长，都会导致已经结束
#                     ## record该序列的 该时刻状态、该时刻动作、下一个状态、动作的奖励、是否完成的
#                     transition_dict['states'].append(state)
#                     transition_dict['actions'].append(action)
#                     transition_dict['next_states'].append(next_state)
#                     transition_dict['rewards'].append(reward)
#                     transition_dict['dones'].append(done)
#                     state = next_state    ## 下一个状态赋值到当前状态
#                     episode_return += reward  ##累加奖励的
#                 return_list.append(episode_return)  ## 训练策略网络的，用一条序列来训练
#                 agent.update(transition_dict)
#                 if (i_episode+1) % 10 == 0:
#                     pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
#                 pbar.update(1)
#     return return_list

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    allimage = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i}') as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'S': [], 'A': [], 'S_': [], 'R': [], 'dones': []}
                s = env.reset()
                if len(s) != 2 * 2:
                    s = s[0]
                done = False
                ## 采样一条序列的
                while not done:
                    if (i_episode + 1) % 10 == 0 and i in [9]:
                        img = env.render()
                        allimage.append(img)
                    # cv2.imshow("CartPole-v1", img)
                    # cv2.waitKey(-1)
                    a = agent.take_action(s)  ##  根据状态采取动作的
                    ##  环境执行动作，并反馈下一个状态、动作的奖励、是否完成、步长太长的，info
                    s_, r, terminated, truncated, info = env.step(a)
                    done = terminated | truncated  ## 终止或者步长太长，都会导致已经结束
                    ## record该序列的 该时刻状态、该时刻动作、下一个状态、动作的奖励、是否完成的
                    transition_dict['S'].append(s)
                    transition_dict['A'].append(a)
                    transition_dict['S_'].append(s_)
                    transition_dict['R'].append(r)
                    transition_dict['dones'].append(done)
                    s = s_  ## 下一个状态赋值到当前状态
                    episode_return += r  ##累加奖励的
                return_list.append(episode_return)
                agent.update(transition_dict)  ## 训练策略网络的，用一条序列来训练
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': f'{num_episodes / 10 * i + i_episode + 1}',
                        'return': f'{np.mean(return_list[-10:]):.3f}'
                    })
                pbar.update(1)
    return return_list, allimage

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated | truncated
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def train_off_policy_agent_withpth(env, agent, num_episodes, replay_buffer, minimal_size, batch_size, epoch, pth, num):
    return_list = []
    epoch = epoch
    allimage = []
    limit = 1000
    for i in range(epoch):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                if len(state)!=2*2:
                    state = state[0]
                done = False
                ## 采样一条序列的
                while not done:
                    if (i_episode + 1) % 10 == 0 and i == epoch - 1 and len(allimage) < limit:
                        img = env.render()
                        allimage.append(img)
                    # cv2.imshow("CartPole-v1", 
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    next_state = next_state.flatten()
                    done = terminated | truncated
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    # https://github.com/guicalare/Img2gif/blob/master/Code/Img2Gif.py
    imageio.mimsave(os.path.join(pth, 'chapter%s.gif'%str(num)), allimage, duration=10)
    return return_list

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:  # 从后往前逆序算，刚好可以算出广义优势估计的值
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)  # 保存每一步的优势估计
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

def train_on_policy_agent_withpth(env, agent, num_episodes, epoch, pth, num):
    return_list = []
    allimage = []
    epoch = epoch
    limit = 1000
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
    imageio.mimsave(os.path.join(pth, 'chapter%s.gif'%str(num)), allimage, duration=10)
    return return_list
