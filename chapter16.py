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
import collections
import itertools
from tqdm import tqdm
from scipy.stats import truncnorm
import imageio

## 交叉熵方式获得当前要采取的动作，不需要显式的策略模型，是选择多条 episodes 内奖励最大的那条 episode 的第一个动作的
class CEM:
    def __init__(self, n_sequence, elite_ratio, fake_env, upper_bound,
                 lower_bound):
        self.n_sequence = n_sequence       ## 很多很多的episodes，>> N，用来sort选择奖励高的episodes
        self.elite_ratio = elite_ratio     ## 实际候选的episodes比例，< 1，最后和 self.n_sequence 相乘
        self.upper_bound = upper_bound     ## 方差的上界
        self.lower_bound = lower_bound     ## 方差的下界
        self.fake_env = fake_env           ## 是虚假环境的

    def optimize(self, state, init_mean, init_var):
        ## 给定初始的 均值 和 方差
        mean, var = init_mean, init_var             ## (26-1,) (26-1,) 共这么多个，向前看26步
        ## 截断标准正态分布的，也就是均值和方差分别是（0, 1）的正态分布，截断的上下界分别是 -2，2
        X = truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(var))
        '''
            >>> a = np.array([0, 1, 2])
            >>> np.tile(a, 2)
            array([0, 1, 2, 0, 1, 2])
        '''
        ## state (3,)----(n_sequence, 3)
        state = np.tile(state, (self.n_sequence, 1)) ## np.tile 是重复 state，也就是重复self._sequence次，方便采样这么多episodes的

        for _ in range(5):
            ##  对均值做上下界处理，用来产生相应的方差
            lb_dist, ub_dist = mean - self.lower_bound, self.upper_bound - mean ## (26-1,) (26-1,) ，向前看26步
            ##  约束以后的方差，使用了上下界均值，也就是最小值  (26-1,) ，向前看26步
            constrained_var = np.minimum(
                np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)),
                var)
            # 生成动作序列 (n_sequence, 26-1)， 向前看26步， 共n_sequence个episodes
            ## 截断到-2，2之间的标准正态分布，采样以后的话，乘上标准差，最后加上均值，也就是服从均值是mean方差是constrained_var的正态分布
            action_sequences = [X.rvs() for _ in range(self.n_sequence)
                                ] * np.sqrt(constrained_var) + mean
            # 计算每条动作序列的累积奖励
            returns = self.fake_env.propagate(state, action_sequences)[:, 0] ## 状态和对应的动作，拿到累积的奖励
            # 选取累积奖励最高的若干条动作序列
            elites = action_sequences[np.argsort(                     ## 对采样episodes使用奖励来sort，然后选择后面较大的episodes
                returns)][-int(self.elite_ratio * self.n_sequence):]
            new_mean = np.mean(elites, axis=0)  ##  根据样本估计总体动作的均值
            new_var = np.var(elites, axis=0)    ##  根据样本估计总体动作的方差
            # 更新动作序列分布
            mean = 0.1 * mean + 0.9 * new_mean  ## EMA update 动作的均值
            var = 0.1 * var + 0.9 * new_var     ## EMA update 动作的方差

        ## 返回要采取的动作
        return mean  ## 几次迭代以后，返回估计的动作均值，此时截断标准正态分布采样，能使得动作episodes的奖励较高的

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Swish(nn.Module):
    ''' Swish激活函数 '''
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

def init_weights(m):
    ''' 初始化模型权重 '''
    ##  就是截断正态分布的
    def truncated_normal_init(t, mean=0.0, std=0.01):
        torch.nn.init.normal_(t, mean=mean, std=std)     ## 用均值=0，标准差是std的正态分布，来初始化模型的权重
        while True:   ## 做截断
            ##  截断区间是 [mean - 2 * std, mean + 2 * std]，也就是拿到不满足条件的布尔值
            cond = (t < mean - 2 * std) | (t > mean + 2 * std)   ## 截断的条件是 标准差的 -2倍到2倍
            if not torch.sum(cond): ## 权重的所有值是否 都被 截断到 区间内
                break  ## 都被截断到区间内，退出循环返回的
            ##  不满足截断条件的地方，再次初始化
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape, device=device), mean=mean, std=std), t)
        return t
    ## 只有 full connect layer层才有权重需要被初始化
    if type(m) == nn.Linear or isinstance(m, FCLayer):
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(m._input_dim)))
        m.bias.data.fill_(0.0)

class FCLayer(nn.Module):
    ''' 集成之后的全连接层 '''
    def __init__(self, input_dim, output_dim, ensemble_size, activation):
        super(FCLayer, self).__init__()
        self._input_dim, self._output_dim = input_dim, output_dim  ## 输入和输出的 dim
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, input_dim, output_dim).to(device))    ## 配置参数的dim
        self._activation = activation
        self.bias = nn.Parameter(torch.Tensor(ensemble_size, output_dim).to(device))  ## 配置参数的dim

    def forward(self, x):
        '''
        torch.bmm：批量的矩阵乘运算
        若输入是(b×n×m)向量, mat2是(b×m×p)向量, 输出是(b×n×p)向量，第一个dim是batch批量
        >>> input = torch.randn(10, 3, 6)
        >>> mat2 = torch.randn(10, 6, 60)
        >>> res = torch.bmm(input, mat2)
        >>> res.size()
        torch.Size([10, 3, 60])
        '''
        return self._activation(torch.add(torch.bmm(x, self.weight), self.bias[:, None, :]))  ## 构造环境模型的 full connect 层

class EnsembleModel(nn.Module):
    ''' 环境模型集成 '''
    def __init__(self,
                 state_dim,
                 action_dim,
                 ensemble_size=5,
                 learning_rate=1e-3):
        super(EnsembleModel, self).__init__()
        # 输出包括均值和方差,因此是状态与奖励维度之和的两倍
        ## 每个状态都服从高斯分布，然后奖励也是服从高斯分布，用来捕捉 偶然不确定性
        self._output_dim = (state_dim + 1) * 2
        self._max_logvar = nn.Parameter((torch.ones(
            (1, self._output_dim // 2)).float() / 2).to(device),
                                        requires_grad=False)         ## 初始化方差的最大值是 1/2
        self._min_logvar = nn.Parameter((-torch.ones(
            (1, self._output_dim // 2)).float() * 10).to(device),
                                        requires_grad=False)         ## 初始化方差的最小值是 -10
        ## 集成的full connect layer
        self.layer1 = FCLayer(state_dim + action_dim, 200, ensemble_size, Swish())
        self.layer2 = FCLayer(200, 200, ensemble_size, Swish())
        self.layer3 = FCLayer(200, 200, ensemble_size, Swish())
        self.layer4 = FCLayer(200, 200, ensemble_size, Swish())
        self.layer5 = FCLayer(200, self._output_dim, ensemble_size, nn.Identity()) ## 最后一层没有激活函数
        self.apply(init_weights)  # 初始化环境模型中的参数
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate) ## 优化器

    ## return_log_var，var是方差，Log_var是方差的log值
    def forward(self, x, return_log_var=False):
        ret = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))   ##  Sequential顺序执行的模型
        ## 【:self._output_dim // 2】  后半部分是均值的
        mean = ret[:, :, :self._output_dim // 2]
        # 在 PETS 算法中,将方差控制在最小值和最大值之间
        '''
        可导的上截断操作
        F.softplus和relu类似，返回值都是大于0的，【self._output_dim // 2:】前半部分是log方差的
        self._max_logvar - ret[:, :, self._output_dim // 2:]，比_max_logvar大的数值会小于0
        F.softplus(self._max_logvar - ret[:, :, self._output_dim // 2:])会将小于0的值置0
        self._max_logvar - F.softplus(self._max_logvar - ret[:, :, self._output_dim // 2:]) 恢复开始的值，但是比_max_logvar大的数值都被截断到_max_logvar
        不同于直接截断操作，这里的 F.softplus 是可导的梯度可以传播
        '''
        logvar = self._max_logvar - F.softplus(self._max_logvar - ret[:, :, self._output_dim // 2:])
        '''
        可导的下截断操作
        logvar - self._min_logvar，比_min_logvar小的数值会小于0
        F.softplus(logvar - self._min_logvar) 会将小于0的值置0
        self._min_logvar + F.softplus(logvar - self._min_logvar) 恢复开始的值，但是比_min_logvar小的数值都被截断到_min_logvar
        不同于直接截断操作，这里的 F.softplus 是可导的梯度可以传播
        '''
        logvar = self._min_logvar + F.softplus(logvar - self._min_logvar)
        ## 返回环境下一步状态和奖励的高斯分布的均值和log方差，拟合偶然不确定性
        ## 也就是下一步状态 用高斯分布采样，奖励 也用高斯分布采样的
        return mean, logvar if return_log_var else torch.exp(logvar)

    def loss(self, mean, logvar, labels, use_var_loss=True):  ## 损失函数的呢
        inverse_var = torch.exp(-logvar)  ## 方差的
        if use_var_loss:
            ## 求均值 和 label的loss，MSE，两者距离越小越好，然后还乘上了方差的逆
            mse_loss = torch.mean(torch.mean(torch.pow(mean - labels, 2) *  ##(6-1, bs, 2*2)---(6-1)
                                             inverse_var,
                                             dim=-1),
                                  dim=-1)
            ## 方差的 Log 值越小越好，所以可以做损失函数
            var_loss = torch.mean(torch.mean(logvar, dim=-1), dim=-1)
            total_loss = torch.sum(mse_loss) + torch.sum(var_loss)
        else:
            ## 求均值 和 label的loss，MSE，两者距离越小越好
            mse_loss = torch.mean(torch.pow(mean - labels, 2), dim=(1, 2))
            total_loss = torch.sum(mse_loss)
        return total_loss, mse_loss

    def train(self, loss):
        self.optimizer.zero_grad()  ## 参数的梯度置 0 
        ##  _max_logvar 和 _min_logvar 都是不反向传播梯度的，requires_grad=False
        loss += 0.01 * torch.sum(self._max_logvar) - 0.01 * torch.sum(self._min_logvar)
        loss.backward()         ## 反向传播求出梯度
        self.optimizer.step()   ## update求出梯度的

class EnsembleDynamicsModel:
    ''' 环境模型集成,加入精细化的训练 '''
    def __init__(self, state_dim, action_dim, num_network=5):
        self._num_network = num_network ## 集成环境模型的个数
        self._state_dim, self._action_dim = state_dim, action_dim  ## 状态的dim，动作的dim
        ## 实例化集成的环境模型
        self.model = EnsembleModel(state_dim,
                                   action_dim,
                                   ensemble_size=num_network)
        self._epoch_since_last_update = 0

    def train(self,
              inputs, ## (200, 2*2)
              labels, ## (200, 2*2)
              batch_size=64,
              holdout_ratio=0.1,
              max_iter=20):
        # 设置训练集与验证集
        '''
        np.random.permutation(10)
        array([1, 7, 4, 3, 0, 9, 2, 5, 8, 6]) # random
        np.random.permutation([1, 4, 9, 12, 15])
        array([15,  1,  9,  4, 12]) # random
        '''
        permutation = np.random.permutation(inputs.shape[0])    ## shuffle 输入
        inputs, labels = inputs[permutation], labels[permutation]    ## shuffle 输入
        num_holdout = int(inputs.shape[0] * holdout_ratio) ## 用来验证的比例，20
        ## (200*(1-0.2), 2*2)
        train_inputs, train_labels = inputs[num_holdout:], labels[num_holdout:]   ## 拿到用来训练的输入和label
        ## (20, 2*2)
        holdout_inputs, holdout_labels = inputs[:num_holdout], labels[:num_holdout]  ## 用来验证网络是否收敛
        holdout_inputs = torch.from_numpy(holdout_inputs).float().to(device)  ##  转torch的tensor
        holdout_labels = torch.from_numpy(holdout_labels).float().to(device)
        ## (6-1, 20, 2*2)
        holdout_inputs = holdout_inputs[None, :, :].repeat([self._num_network, 1, 1])              ## 重复很多次，每个环境网络拿到一份输入的拷贝
        holdout_labels = holdout_labels[None, :, :].repeat([self._num_network, 1, 1])              ## 重复很多次，每个环境网络拿到一份label的拷贝

        # 保留最好的结果
        self._snapshots = {i: (None, 1e10) for i in range(self._num_network)} ## 每个环境网络的结果

        ## 训练的epoch计数，不会自动停止，需要 break 才可以
        for epoch in itertools.count():
            # 定义每一个网络的train数据
            '''
            np.random.permutation(train_inputs.shape[0])  : shuffle 输入数据的 index
            for _ in range(self._num_network)   : 多少个环境网络，就重复多少次的
            因每个网络都做了一次permutation shuffle，所以每个网络的输入数据的sequence都是不相同的，
            虽然总体的训练数据相同，但是因sequence不同导致了每个batch的输入都不相同。
            np.vstack：最后 vstack起来的，也就是最后的输入 index
            下面的输出train_index的dim是：(20, 6)
            train_index = np.vstack([
                            np.random.permutation(6)
                            for _ in range(20)
                        ])
            train_index 的 dim 是（ _num_network，train_inputs.shape[0]）
            '''
            ## (6-1, 200-20)
            train_index = np.vstack([
                np.random.permutation(train_inputs.shape[0])
                for _ in range(self._num_network)
            ])
            # 所有真实数据都用来train
            for batch_start_pos in range(0, train_inputs.shape[0], batch_size):  ## 每次输入的数量是 batch_size 个
                ## train_index的第一个dim是 环境网络的个数，第二个dim才是 数据的个数 （6-1，bs）
                batch_index = train_index[:, batch_start_pos : batch_start_pos + batch_size]  ## (num_network, train_input.shape[0])
                train_input = torch.from_numpy(train_inputs[batch_index]).float().to(device)  ## 拿到输入的数据，（6-1，bs, 2*2）
                train_label = torch.from_numpy(train_labels[batch_index]).float().to(device)  ## 输入的label，（6-1，bs, 2*2）
                ## train 集成起来的多个环境模型 （6-1，bs, 2*2） （6-1，bs, 2*2）
                mean, logvar = self.model(train_input, return_log_var=True) 
                loss, _ = self.model.loss(mean, logvar, train_label)  ## 算 loss
                self.model.train(loss)  ## train

            with torch.no_grad():  ## 不算梯度的
                mean, logvar = self.model(holdout_inputs, return_log_var=True)  ## 其他的数据前向传播 （6-1，20, 2*2）（6-1，20, 2*2）
                _, holdout_losses = self.model.loss(mean,
                                                    logvar,
                                                    holdout_labels,
                                                    use_var_loss=False)   ## 算出loss的，只算均值损失，不算方差的损失
                holdout_losses = holdout_losses.cpu()
                break_condition = self._save_best(epoch, holdout_losses)  ## 根据损失的下降程度，决定是否保存模型
                if break_condition or epoch > max_iter:  # 结束训练
                    break

    def _save_best(self, epoch, losses, threshold=0.1):
        updated = False                    ## 是否 update 模型的
        for i in range(len(losses)):  
            current = losses[i]            ## 当前的损失
            _, best = self._snapshots[i]   ## 保存的损失
            improvement = (best - current) / best  ## 损失下降的比例
            if improvement > threshold:    ## 下降的比例大于 threshold
                self._snapshots[i] = (epoch, current)  ## 保存当前的epoch和损失
                updated = True ## 损失update过了的
        ## 用来做train的终止条件的，若是距离上次 update 已经过去了6个epoch，那么就可以终止train
        self._epoch_since_last_update = 0 if updated else self._epoch_since_last_update + 1
        return self._epoch_since_last_update > 5

    def predict(self, inputs, batch_size=64): ## (n_sequence, 2*2)
        mean, var = [], []
        ## 使用多个环境模型来 predict 下个状态和奖励分布的 均值和方差
        for i in range(0, inputs.shape[0], batch_size):
            input = torch.from_numpy(
                inputs[i:min(i +
                             batch_size, inputs.shape[0])]).float().to(device)
            cur_mean, cur_var = self.model(input[None, :, :].repeat(  ## input[None, :, :].repeat([self._num_network, 1, 1])  (6-1, n_sequence, 2*2)
                [self._num_network, 1, 1]),
                                           return_log_var=False)
            mean.append(cur_mean.detach().cpu().numpy()) ## (6-1, n_sequence, 2*2)
            var.append(cur_var.detach().cpu().numpy()) ## (6-1, n_sequence, 2*2)
        return np.hstack(mean), np.hstack(var)

## 构造的虚拟环境，不是真实环境的，虚拟环境只在CEM交叉熵方式选取动作时需要用到，其他地方用不到
class FakeEnv:
    def __init__(self, model):
        ## 集成环境模型
        self.model = model

    def step(self, obs, act):
        inputs = np.concatenate((obs, act), axis=-1)  ## 拼接状态和动作 （n_sequence, 2*2）
        ## 虚拟环境来predict 下个状态和奖励的 均值和方差
        ensemble_model_means, ensemble_model_vars = self.model.predict(inputs) ##  （6-1, n_sequence, 2*2）（6-1, n_sequence, 2*2）
        '''
        下个状态的均值 加上 当前的状态，所以集成环境模型，返回的均值其实是当前状态的残差
        需要加上当前状态，才是真正的下一个状态的均值
        '''
        ensemble_model_means[:, :, 1:] += obs.numpy()
        ensemble_model_stds = np.sqrt(ensemble_model_vars) ## 算标准差的
        ## 标准正态分布采样，然后乘标准差加均值，变到常规正态分布 (ensemble_model_means, ensemble_model_stds**2) 采样
        ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds  ## （6-1, n_sequence, 2*2）

        num_models, batch_size, _ = ensemble_model_means.shape ## 均值的shape  （6-1, n_sequence, 2*2）
        models_to_use = np.random.choice(
            [i for i in range(self.model._num_network)], size=batch_size)  ## 随机选择环境模型 ##(batch_size,)
        batch_inds = np.arange(0, batch_size)
        samples = ensemble_samples[models_to_use, batch_inds] ## 拿到实际的采样值  ##(batch_size,2*2)
        rewards, next_obs = samples[:, :1], samples[:, 1:]    ## 拿出 奖励值，以及 下一个状态  (batch_size, 1)  (batch_size, 2*2-1)
        return rewards, next_obs

    ##  使用了 状态 + 动作 episodes，然后不保存梯度，前向算出累积奖励的
    def propagate(self, obs, actions): ## （n_sequence, 3） （n_sequence, 26）  ，向前看26步
        with torch.no_grad():  ##  不保存梯度的
            obs = np.copy(obs) ##  复制状态的 （n_sequence, 3）
            total_reward = np.expand_dims(np.zeros(obs.shape[0]), axis=-1) ## 初始化累积奖励 （n_sequence，1）
            obs, actions = torch.as_tensor(obs), torch.as_tensor(actions)  ## 向量化
            for i in range(actions.shape[1]):  ## 每个episodes的长度 H，26，也就是这个状态后续的节点个数，也就是向前看26步 =actions.shape[1]
                action = torch.unsqueeze(actions[:, i], 1)  ## 加dim，方便后续运算的，拿到第i个时刻的节点动作  (n_sequence, 1)
                rewards, next_obs = self.step(obs, action)  ## 虚拟环境根据 状态+动作 来执行动作，并返回奖励和下一个状态 (n_sequence, 1)  (n_sequence, 2*2 - 1)
                total_reward += rewards  ## 累积奖励的 
                obs = torch.as_tensor(next_obs) ## 向量化
            return total_reward ## 返回累积奖励的

## 保存历史数据，也就是回放池，用来train模型的
class ReplayBuffer:
    def __init__(self, capacity):
        ## 回放池
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        ## 加入到回放池
        self.buffer.append((state, action, reward, next_state, done))

    def size(self):
        ## 返回回放池内个数的
        return len(self.buffer)

    def return_all_samples(self):
        ## 返回保存的所有历史数据，并转置，每列都是不同的
        all_transitions = list(self.buffer)
        state, action, reward, next_state, done = zip(*all_transitions)
        return np.array(state), action, reward, np.array(next_state), done

class PETS:
    ''' PETS算法 '''
    def __init__(self, env, replay_buffer, n_sequence, elite_ratio,
                 plan_horizon, num_episodes):
        self._env = env     ## 环境的
        self._env_pool = ReplayBuffer(buffer_size)   ## 保存历史数据，回放池

        obs_dim = env.observation_space.shape[0]     ## 状态的dim
        self._action_dim = env.action_space.shape[0] ## 动作的dim
        self._model = EnsembleDynamicsModel(obs_dim, self._action_dim) ## 集成环境模型的
        self._fake_env = FakeEnv(self._model)        ## 实例化虚拟环境，虚拟环境只在CEM交叉熵方式选取动作时需要用到，其他地方用不到
        self.upper_bound = env.action_space.high[0]  ## 动作的最大值
        self.lower_bound = env.action_space.low[0]   ## 动作的最小值

        ## 实例化交叉熵方式，向前看几步用来给出下一个动作，取代了策略网络
        self._cem = CEM(n_sequence, elite_ratio, self._fake_env, self.upper_bound, self.lower_bound)
        self.plan_horizon = plan_horizon ## 指定向前看多少步
        self.num_episodes = num_episodes

    def train_model(self):
        env_samples = self._env_pool.return_all_samples()  ## 返回回放池内所有的历史数据
        obs = env_samples[0]  ## 所有历史状态
        actions = np.array(env_samples[1]) ## 所有历史动作
        rewards = np.array(env_samples[2]).reshape(-1, 1) ## 所有历史奖励
        next_obs = env_samples[3] ## 所有下一步的状态
        inputs = np.concatenate((obs, actions), axis=-1) ## 输入
        ## label是奖励，以及下一步的状态 减去 当前状态得到的残差
        labels = np.concatenate((rewards, next_obs - obs), axis=-1) ## 标签
        self._model.train(inputs, labels) ## train

    def mpc(self, index, num_episode):
        ## np.tile重复均值的
        allimage = []
        mean = np.tile((self.upper_bound + self.lower_bound) / 2.0, self.plan_horizon)            ## 指定向前看多少步均值
        ## np.tile重复方差的
        var = np.tile(np.square(self.upper_bound - self.lower_bound) / 16, self.plan_horizon)     ## 指定向前看多少步方差
        ## 环境重置的，完成标志初始化False，返回的episode
        obs, done, episode_return = self._env.reset(), False, 0
        if len(obs)!=2*2-1:
            obs = obs[0]
        while not done:
            if index==num_episode - 1:
                img = self._env.render()
                allimage.append(img)
            ## 交叉熵方式 来选择 奖励较高的 动作episodes，截断标准正态分布，然后采样算奖励，用奖励较高的episodes来update分布的均值和方差
            ## 样本估计总体的，CEM根据当前状态，均值和方差，返回要采取的动作，不使用策略网络返回动作
            ## (3,) (26-1,) (26-1,)
            actions = self._cem.optimize(obs, mean, var) ## (26-1,)，也就是选择的累加奖励最大的episode的向前看的动作
            action = actions[:self._action_dim]  # 选取第一个动作
            ##  环境执行动作，并反馈下一个状态、动作的奖励、是否完成、步长太长的，info
            next_obs, reward, terminated, truncated, info = self._env.step(action)
            done = terminated | truncated       ## 终止或者步长太长，都会导致已经结束
            # next_obs, reward, done, _ = self._env.step(action)  ## 根据动作返回下一个状态、动作的奖励
            ## （当前的状态，当前的动作，奖励，下一个状态，是否完成的）加入到回放池内
            self._env_pool.add(obs, action, reward, next_obs, done)  ## CEM+虚拟的环境选择动作并和真实环境交互的数据
            obs = next_obs
            episode_return += reward
            mean = np.concatenate([
                np.copy(actions)[self._action_dim:],
                np.zeros(self._action_dim)
            ])  ## 使用当前动作来做均值
        if index == num_episode - 1:
            # https://github.com/guicalare/Img2gif/blob/master/Code/Img2Gif.py
            pth = r'C:\Users\10696\Desktop\access\Hands-on-RL'
            imageio.mimsave(os.path.join(pth, 'chapter%s.gif'%str("16")), allimage, duration=10)
        return episode_return

    ## 探索环境的，使用真实环境，来获得真实的交互数据
    def explore(self):
        obs, done, episode_return = self._env.reset(), False, 0 ## 重置环境的
        if len(obs)!=2*2-1:
            obs = obs[0]
        while not done:
            action = self._env.action_space.sample() ## 真实环境随机采取动作
            ##  环境执行动作，并反馈下一个状态、动作的奖励、是否完成、步长太长的，info
            next_obs, reward, terminated, truncated, info = self._env.step(action)
            done = terminated | truncated       ## 终止或者步长太长，都会导致已经结束
            # next_obs, reward, done, _ = self._env.step(action)
            self._env_pool.add(obs, action, reward, next_obs, done) ## 加入到回放池  和真实环境交互的数据
            obs = next_obs
            episode_return += reward
        return episode_return ## 返回累积奖励的

    def train(self):
        return_list = []
        ## 直接返回奖励的
        explore_return = self.explore()  # 先进行随机策略的探索来收集一条序列的数据
        print('episode: 1, return: %d' % explore_return)
        return_list.append(explore_return)
        
        for i_episode in range(self.num_episodes - 1):  ## 指定episode的数量
            self.train_model()                          ## 使用回放池内的数据 train 集成的环境模型
            episode_return = self.mpc(i_episode, self.num_episodes - 1)                 ## 使用CEM交叉熵的方式，来获取动作的，并得到相应的episode
            return_list.append(episode_return)          ## 加入MPC方式episode的累积奖励
            print('episode: %d, return: %d' % (i_episode + 2, episode_return))
        return return_list                              ## 返回累积奖励的列表

buffer_size = 100000
n_sequence = 50
elite_ratio = 0.2
plan_horizon = 25
num_episodes = 10
env_name = 'Pendulum-v1'
env = gym.make(env_name, render_mode="rgb_array")

replay_buffer = ReplayBuffer(buffer_size)            ## 实例化缓冲回放池，存放历史数据
## probabilistic ensembles with trajectory sampling，带有轨迹采样的概率集成
pets = PETS(env, replay_buffer, n_sequence, elite_ratio, plan_horizon, num_episodes)
return_list = pets.train()

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PETS on {}'.format(env_name))
plt.show()

# episode: 1, return: -1062
# episode: 2, return: -1257
# episode: 3, return: -1792
# episode: 4, return: -1225
# episode: 5, return: -248
# episode: 6, return: -124
# episode: 7, return: -249
# episode: 8, return: -269
# episode: 9, return: -245
# episode: 10, return: -119