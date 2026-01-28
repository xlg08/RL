# -*- coding: utf-8 -*-
from tqdm import tqdm
import gymnasium as gym  # Gymnasium 环境库（推荐用于新项目）
import torch  # PyTorch 深度学习框架
import torch.nn.functional as F  # 神经网络函数模块
import numpy as np  # 数值计算库
import matplotlib.pyplot as plt  # 数据可视化库


# 定义策略网络 PolicyNet（继承自 PyTorch Module）
class PolicyNet(torch.nn.Module):
    """
    策略网络：用于输出状态 s 下每个动作的概率分布（使用 softmax）。
    输入：state_dim -> 状态维度
         hidden_dim -> 隐藏层维度
         action_dim -> 动作维度
    输出：softmax 归一化的动作概率
    """

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # 第一层全连接
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)  # 第二层全连接

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 使用 ReLU 激活函数
        return F.softmax(self.fc2(x), dim=1)  # 输出 softmax 动作概率分布


# 定义价值网络 ValueNet（用于估计状态价值）
class ValueNet(torch.nn.Module):
    """
    价值网络：用于估计当前状态的价值（即状态的 V 值）。
    输入：state_dim -> 状态维度
         hidden_dim -> 隐藏层维度
    输出：状态的 V 值
    """

    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# Actor-Critic 算法实现类
class ActorCritic:
    """
    Actor-Critic 算法实现类，包含策略网络、价值网络、更新规则等。
    """

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)  # 策略网络
        self.critic = ValueNet(state_dim, hidden_dim).to(device)  # 价值网络
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)  # 策略优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)  # 价值优化器
        self.gamma = gamma  # 折扣因子
        self.device = device  # 设备（CPU or GPU）

    def take_action(self, state):
        """
        根据当前状态选择动作（使用策略网络采样一个动作）。
        参数：
            state: 当前状态
        返回：
            action: 选择的动作编号
        """
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        """
        使用经验数据更新 Actor 和 Critic 网络。
        参数：
            transition_dict: 包含 states, actions, rewards, next_states, dones 的字典
        """
        # 将经验数据转换为 tensor 并移动到指定设备
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

        # 计算时序差分目标 (TD Target)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)

        # 计算 TD 误差（用于策略梯度更新）
        td_delta = td_target - self.critic(states)

        # 获取 log(prob) 并乘以 TD delta
        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = torch.mean(-log_probs * td_delta.detach())

        # 计算价值网络的均方误差损失
        critic_loss = torch.mean(
            F.mse_loss(self.critic(states), td_target.detach()))

        # 反向传播并更新参数
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()  # 更新策略网络
        critic_loss.backward()  # 更新价值网络
        self.actor_optimizer.step()
        self.critic_optimizer.step()


def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state, _ = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)  # 执行动作
                    done = terminated or truncated  # 合并两个条件作为 done 标志

                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


# 设置超参数
actor_lr = 1e-3  # 策略网络学习率
critic_lr = 1e-2  # 价值网络学习率
num_episodes = 1000  # 总共训练的回合数
hidden_dim = 128  # 网络隐藏层大小
gamma = 0.98  # 折扣因子
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 创建环境
env_name = 'CartPole-v0'
env = gym.make(env_name)  # 初始化环境
env.action_space.seed(0)  # 设置动作空间种子
torch.manual_seed(0)  # 设置 PyTorch 随机种子

# 获取状态空间和动作空间维度
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 实例化 Actor-Critic 算法对象
agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                    gamma, device)

# 记录每个 episode 的回报
return_list = []

# 开始训练
return_list = train_on_policy_agent(env, agent, num_episodes)

# 绘图展示训练过程中的回报曲线
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Actor-Critic on {}'.format(env_name))
plt.show()

# 绘制滑动平均回报曲线
mv_return = moving_average(return_list, 9)  # 9点滑动平均
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Actor-Critic on {}'.format(env_name))
plt.show()
