# -*- coding: utf-8 -*-
# 引入必要的库
# import gym  # OpenAI Gym 环境库
import gymnasium as gym
import torch  # PyTorch 深度学习框架
import torch.nn.functional as F  # 神经网络函数模块
import numpy as np  # 数值计算库
import matplotlib.pyplot as plt  # 数据可视化库
from tqdm import tqdm  # 显示进度条的库


# 定义策略网络 PolicyNet
class PolicyNet(torch.nn.Module):
    """
    策略网络，用于输出状态下的动作概率分布。
    输入：state_dim -> 状态维度
         hidden_dim -> 隐藏层维度
         action_dim -> 动作维度
    输出：softmax 归一化的动作概率分布
    """

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # 第一层全连接
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)  # 第二层全连接

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 使用 ReLU 激活函数
        return F.softmax(self.fc2(x), dim=1)  # 使用 softmax 输出动作概率分布


# REINFORCE 算法实现类
class REINFORCE:
    """
    REINFORCE 算法：基于策略梯度的强化学习方法。
    """

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, device):
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        # 使用 Adam 优化器更新策略网络参数
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.device = device  # 设备选择（CPU 或 GPU）

    def take_action(self, state):
        """
        根据当前状态采样一个动作（使用分类分布进行随机采样）。
        """
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.policy_net(state)  # 获取当前状态的动作概率分布
        action_dist = torch.distributions.Categorical(probs)  # 构造分类分布
        action = action_dist.sample()  # 采样动作
        return action.item()  # 返回动作编号

    def update(self, transition_dict):
        """
        使用轨迹数据对策略网络进行更新。
        """
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0  # 初始化回报 G_t
        self.optimizer.zero_grad()  # 清空梯度

        # 从后往前遍历每一步的轨迹数据（MC 方法）
        for i in reversed(range(len(reward_list))):
            reward = reward_list[i]
            state = torch.tensor([state_list[i]], dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)

            probs = self.policy_net(state)  # 获取状态对应的动作概率
            log_prob = torch.log(probs.gather(1, action))  # 计算 log π(a|s)

            G = self.gamma * G + reward  # 累积折扣回报
            loss = -log_prob * G  # 策略梯度损失函数
            loss.backward()  # 反向传播计算梯度

        self.optimizer.step()  # 更新网络参数（梯度下降）


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


# 设置超参数
learning_rate = 1e-3  # 学习率
num_episodes = 1000  # 总共训练的回合数
hidden_dim = 128  # 网络隐藏层大小
gamma = 0.98  # 折扣因子
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 创建环境
env_name = "CartPole-v0"  # CartPole 环境
env = gym.make(env_name)  # 初始化环境
env.action_space.seed(0)  # 固定种子以保证实验可复现
torch.manual_seed(0)  # 固定 PyTorch 的随机种子

# 获取状态空间和动作空间维度
state_dim = env.observation_space.shape[0]  # 状态维度
action_dim = env.action_space.n  # 动作数量

# 实例化 REINFORCE 算法对象
agent = REINFORCE(state_dim, hidden_dim, action_dim, learning_rate, gamma, device)

# 记录每个 episode 的回报
return_list = []

# 开始训练
for i in range(10):  # 分批次显示进度条
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):  # 每个批次的 episode 数量
            episode_return = 0  # 当前 episode 的总回报
            transition_dict = {  # 存储单次 episode 的轨迹数据
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': []
            }
            state, _ = env.reset()  # 环境重置
            done = False  # 是否结束标志

            # 运行单个 episode
            while not done:
                action = agent.take_action(state)  # 采样动作
                next_state, reward, terminated, truncated, _ = env.step(action)  # 执行动作
                done = terminated or truncated  # 合并两个条件作为 done 标志

                # 存储轨迹数据
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)

                state = next_state  # 更新状态
                episode_return += reward  # 累计奖励

            return_list.append(episode_return)  # 保存 episode 回报
            agent.update(transition_dict)  # 更新策略网络

            # 每 10 个 episode 打印一次平均回报
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return': '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)  # 更新进度条

# 绘图展示训练过程中的回报曲线
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('REINFORCE on {}'.format(env_name))
plt.show()

# 绘制滑动平均回报曲线
mv_return = moving_average(return_list, 9)  # 9点滑动平均
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('REINFORCE on {}'.format(env_name))
plt.show()
