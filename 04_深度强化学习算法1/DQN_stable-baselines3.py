# -*- coding: utf-8 -*-

import gymnasium as gym  # Gymnasium 环境库
from stable_baselines3 import DQN  # Stable Baselines3 中的 DQN 算法
from stable_baselines3.common.callbacks import BaseCallback  # 自定义回调函数基类
import matplotlib.pyplot as plt  # 数据可视化库
import numpy as np  # 数值计算库


# 自定义回调函数用于记录每个 episode 的回报
class RewardLogger(BaseCallback):
    """
    BaseCallback 是 Stable Baselines3 提供的回调函数基类，它定义了回调函数的基本结构和接口规范
    所以需要继承并实现必要的方法
    在训练过程中记录每个 episode 的总奖励。
    """

    def __init__(self, verbose=0):
        super(RewardLogger, self).__init__(verbose)
        self.episode_rewards = []  # 存储每个 episode 的总奖励
        self.current_episode_reward = 0  # 当前 episode 的累计奖励

    def _on_step(self) -> bool:
        """
        _on_step(): 每个时间步都会调用，必须实现
        每一步都会调用此方法，累加当前 episode 的奖励。
        如果 episode 结束，则保存当前 episode 的总奖励。
        """
        # self.locals 是 BaseCallback 类中的一个重要属性，在每次调用 _on_step() 方法时自动更新
        # self.locals 是一个字典，包含当前时间步的信息，如状态、动作、奖励、是否结束等
        # self.locals['rewards'] 是一个列表，包含当前时间步的奖励
        # self.locals['dones'] 是一个列表，包含当前时间步的结束标志
        reward = self.locals['rewards'][0]  # 获取当前步的奖励
        self.current_episode_reward += reward

        # 如果当前 episode 结束
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0  # 重置当前 episode 奖励

        return True  # 返回 True 表示继续训练


# 创建环境和模型
env = gym.make('CartPole-v0')  # 使用 CartPole-v0 环境

# 初始化 DQN 模型
model = DQN(
    'MlpPolicy',  # 使用多层感知机策略（适用于低维状态空间）
    env,  # 绑定的 Gymnasium 环境
    learning_rate=2e-3,  # 学习率
    batch_size=64,  # 训练时每次采样的批次大小
    buffer_size=20000,  # 经验回放缓冲区容量
    learning_starts=1000,  # 开始学习前的随机探索步数
    target_update_interval=100,  # 更新目标网络的频率（步数）
    gamma=0.98,  # 折扣因子
    verbose=1  # 打印训练日志
)

# 初始化回调函数
reward_logger = RewardLogger()

# 开始训练模型，设置总步数为 10000 步，具体的 episode 数量是由环境中自动完成的。
# 每当一次 episode 结束（即 done=True），就会开始一个新的 episode，直到累计达到 10000 步为止。
model.learn(total_timesteps=10000, callback=reward_logger)

# 绘制回报曲线
plt.figure(figsize=(12, 6))
print(reward_logger.episode_rewards)
print(type(reward_logger.episode_rewards))
print(len(reward_logger.episode_rewards))
plt.plot(reward_logger.episode_rewards, label='Episode Reward')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DQN Training Performance on CartPole-v0')
plt.grid(True)

# 添加移动平均线（窗口大小为10）
window_size = 10
moving_avg = np.convolve(reward_logger.episode_rewards,
                         np.ones(window_size) / window_size,
                         mode='valid')

# 移动平均线从第 window_size 个 episode 开始绘制
plt.plot(range(window_size - 1, len(reward_logger.episode_rewards)),
         moving_avg,
         'r-',
         linewidth=2,
         label=f'{window_size}-episode Moving Avg')

plt.legend()
plt.tight_layout()

# 保存图像到本地
plt.savefig('dqn_cartpole_rewards.png', dpi=300)

# 显示图像
plt.show()
