# -*- coding: utf-8 -*-
import random
import gymnasium as gym
import numpy as np  # 数值计算库
import collections  # 提供常用数据结构（如 deque）
from tqdm import tqdm  # 显示进度条的库
import torch  # PyTorch 深度学习框架
import torch.nn.functional as F  # 神经网络函数模块
import matplotlib.pyplot as plt  # 数据可视化库


class ReplayBuffer:
    """
    经验回放池：用于存储智能体与环境交互的历史经验，并从中随机采样小批量数据进行训练。
    """

    def __init__(self, capacity):
        """ deque是一个类似于列表（list - like）的序列，但针对在其端点附近进行数据访问进行了优化
        deque（全称double - ended queue，这意味着你可以在序列的左端和右端都进行高效的添加和删除操作。
        它与普通列表（list）最大的区别在于性能。对于列表，在开头插入或删除元素（insert(0, v)或
        pop(0)）的时间复杂度是O(n)，因为需要移动所有其他元素。而deque在两端进行添加和删除操作的时间复杂度都是
        O(1)，非常高效"""
        self.buffer = collections.deque(maxlen=capacity)  # 使用双端队列存储经验

    def add(self, state, action, reward, next_state, done):
        """
        添加一条经验到缓冲池中。
        参数：
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否结束标志
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        随机采样一个批次的经验数据。
        参数：
            batch_size: 批量大小
        返回：
            states: 状态数组
            actions: 动作数组
            rewards: 奖励数组
            next_states: 下一状态数组
            dones: 结束标志数组
        """
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        """
        获取当前缓冲池中存储的经验数量。
        """
        return len(self.buffer)


class Qnet(torch.nn.Module):
    """
    只有一层隐藏层的Q网络，用于估计每个动作的价值。
    输入：state_dim -> 状态维度
         hidden_dim -> 隐藏层维度
         action_dim -> 动作维度
    输出：每个动作的Q值
    """

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # 第一层全连接
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)  # 第二层全连接

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 使用 ReLU 激活函数
        return self.fc2(x)  # 输出每个动作的 Q 值


class DQN:
    """
    DQN 算法实现类，包含策略网络、目标网络、更新规则等。
    """

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)  # Q网络
        self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)  # 目标网络
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)  # Adam优化器
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率（epsilon-greedy）
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 更新计数器
        self.device = device  # 设备（CPU or GPU）

    def take_action(self, state):
        """
        使用 epsilon-greedy 策略选择动作。
        参数：
            state: 当前状态
        返回：
            action: 选择的动作编号
        """
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        """
        使用经验数据更新 Q 网络。
        参数：
            transition_dict: 包含 states, actions, rewards, next_states, dones 的字典
        """
        # 将经验数据转换为 tensor 并移动到指定设备
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 计算当前 Q 值:根据 actions 中对应的索引值,从 q_values 中提取该动作对应的 Q 值,最终得到一个形状为 [batch_size, 1] 的张量
        """
        示例：# Q网络对所有动作的预测值 (shape: [2, 3])
            # 状态1: 动作0的Q=0.1, 动作1的Q=0.5, 动作2的Q=0.3
            # 状态2: 动作0的Q=0.2, 动作1的Q=0.4, 动作2的Q=0.6
            q_net_output = torch.tensor([[0.1, 0.5, 0.3],
                                         [0.2, 0.4, 0.6]])
            
            # 实际采取的动作索引 (shape: [2, 1])
            actions = torch.tensor([[1],  # 在状态1采取了动作1
                                    [2]]) # 在状态2采取了动作2
            
            # 使用gather提取对应动作的Q值
            gather方法：按照“索引张量”在指定维度上，从原张量中批量“取值”并重新组成一个新张量
            q_values = q_net_output.gather(1, actions)
            print(q_values)
            # 输出: tensor([[0.5],  # 对应状态1的动作1的Q值
            #              [0.6]]) # 对应状态2的动作2的Q值
        """
        q_values = self.q_net(states).gather(1, actions)        # 取出一维中的值

        # 使用目标网络计算下一状态的最大 Q 值
        """
        target_q_net网络结构与q_net一样，输出形状为[batch_size, action_dim]，表示每个状态下所有动作的Q值
        .max(1)是在第一个维度（动作维度）寻找最大值，返回一个元组：（最大值，对应的索引）
        [0]表示取元组中最大值部分，丢弃索引
        .view(-1, 1)，将结果重塑为列向量形式，形状为 [batch_size, 1]
        """
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)

        # TD error 目标
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        # 计算均方误差损失
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))

        # 反向传播更新参数
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        # 定期更新目标网络
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1


# 设置超参数
lr = 2e-3  # 学习率
num_episodes = 500  # 总共训练的回合数
hidden_dim = 128  # 网络隐藏层大小
gamma = 0.98  # 折扣因子
epsilon = 0.01  # 探索率
target_update = 10  # 目标网络更新频率
buffer_size = 10000  # 经验回放缓冲区大小
minimal_size = 500  # 开始训练前需要的最小经验数量
batch_size = 64  # 每次训练使用的经验批大小
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 创建环境
env_name = 'CartPole-v0'
env = gym.make(env_name)
random.seed(0)
np.random.seed(0)
env.action_space.seed(0)
torch.manual_seed(0)

# 初始化经验回放缓冲区
replay_buffer = ReplayBuffer(buffer_size)

# 获取状态空间和动作空间维度
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 实例化 DQN 算法对象
agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)

# 记录每个 episode 的回报
return_list = []

# 开始训练
for i in range(10):  # 分批次显示进度条
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):  # 每个批次的 episode 数量
            episode_return = 0  # 当前 episode 的总回报
            state, _ = env.reset()  # 环境重置
            done = False  # 是否结束标志

            # 运行单个 episode (轨迹  ==>   包含状态、动作、奖励...)
            while not done:
                action = agent.take_action(state)  # 采样动作
                next_state, reward, terminated, truncated, _ = env.step(action)  # 执行动作
                done = terminated or truncated  # 合并两个条件作为 done 标志

                # 存储轨迹数据
                replay_buffer.add(state, action, reward, next_state, done)

                state = next_state  # 更新状态
                episode_return += reward  # 累计奖励

                # 当 buffer 数据足够时，开始训练
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    agent.update(transition_dict)
            return_list.append(episode_return)  # 保存 episode 回报
            # 每 10 个 episode 打印一次平均回报
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return': '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)  # 更新进度条


def moving_average(a, window_size):
    """
    计算数组的滑动平均值
    Args:
        a (array-like): 输入数组
        window_size (int): 滑动窗口大小

    Returns:
        numpy.ndarray: 滑动平均后的数组
    """
    # 计算累积和，在数组开头插入0以便计算
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    # 计算中间部分的滑动平均值
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    # 处理边界情况
    r = np.arange(1, window_size - 1, 2)
    # 计算起始部分的滑动平均值
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    # 计算结束部分的滑动平均值
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    # 拼接三部分结果
    return np.concatenate((begin, middle, end))


# 绘图展示训练过程中的回报曲线
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()

# 绘制滑动平均回报曲线
mv_return = moving_average(return_list, 9)  # 9点滑动平均
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()
