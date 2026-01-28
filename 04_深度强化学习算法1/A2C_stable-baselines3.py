import gymnasium as gym  # Gymnasium 环境库
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env


# 自定义回调函数：记录每个episode的回报
class RewardLogger(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLogger, self).__init__(verbose)
        self.episode_rewards = []  # 存储所有episode的累计奖励
        self.current_episode_reward = 0  # 当前episode的累计奖励

    def _on_step(self) -> bool:
        reward = self.locals['rewards'][0]  # 获取当前步骤的奖励
        self.current_episode_reward += reward

        # 若episode结束，记录并重置累计奖励
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
        return True


# 创建环境和模型
env = gym.make('CartPole-v0')  # 使用 CartPole-v0 环境

# 初始化A2C模型
model = A2C(
    policy='MlpPolicy',
    env=env,
    learning_rate=7e-4,  # 学习率（A2C默认值）
    n_steps=5,  # 每次更新前的步数
    gamma=0.99,  # 折扣因子
    gae_lambda=1.0,  # GAE参数
    ent_coef=0.0,  # 熵系数（0表示无熵正则化）
    verbose=1,  # 打印训练日志
    tensorboard_log="./a2c_logs"  # TensorBoard日志目录[1](@ref)
)

# 初始化回调函数
reward_logger = RewardLogger()

# 开始训练模型，设置总步数为 10000 步，具体的 episode 数量是由环境中自动完成的。
# 每当一次 episode 结束（即 done=True），就会开始一个新的 episode，直到累计达到 10000 步为止。
model.learn(total_timesteps=10000, callback=reward_logger)

# 绘制回报曲线
plt.figure(figsize=(12, 6))
plt.plot(reward_logger.episode_rewards, alpha=0.6, label='Episode Reward')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('A2C Training Performance on CartPole-v1')
plt.grid(True)

# 添加10-episode移动平均线（平滑波动）
window_size = 10
moving_avg = np.convolve(
    reward_logger.episode_rewards,
    np.ones(window_size) / window_size,
    mode='valid'
)
plt.plot(
    range(window_size - 1, len(reward_logger.episode_rewards)),
    moving_avg,
    'r-',
    linewidth=2,
    label=f'{window_size}-Episode Moving Avg'
)

plt.legend()
plt.tight_layout()
plt.show()
