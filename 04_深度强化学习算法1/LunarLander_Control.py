# -*- coding: utf-8 -*-
import gymnasium as gym  # Gymnasium 环境库
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    """
    主程序入口：创建环境、训练模型、测试模型并可视化训练过程
    """

    # 创建单个 LunarLander-v3 环境
    env = gym.make('LunarLander-v3')  # 使用 LunarLander-v3 环境（v2 已弃用）

    # 定义策略网络结构（两层隐藏层，每层128个神经元）
    policy_kwargs = dict(
        net_arch=[128, 128]  # 两层隐藏层，每层128个神经元
    )

    # 初始化 PPO 模型
    model = PPO(
        "MlpPolicy",  # 使用多层感知机策略（适用于低维状态空间）
        env,  # 绑定 Gymnasium 环境
        policy_kwargs=policy_kwargs,
        gamma=0.99,  # 折扣因子 γ，控制未来奖励的重要性
        n_steps=2048,  # 每次更新前的步数（经验缓冲区大小）
        batch_size=64,  # 每次优化使用的批量大小
        ent_coef=0.01,  # 熵系数（鼓励探索）
        verbose=1,  # 打印训练信息
        tensorboard_log="./lunar_log/"  # TensorBoard 日志路径
    )

    # 添加评估回调函数（可选）：定期评估模型性能并保存最佳模型
    eval_callback = EvalCallback(
        env,
        # eval_freq=10_000,  # 每1万步评估一次
        eval_freq=100,  # 每1万步评估一次
        best_model_save_path="best_model/",  # 最佳模型保存路径
        deterministic=True,  # 使用确定性策略进行评估
        render=False  # 评估时不渲染画面
    )


    # 自定义回调类：记录每个 episode 的回报值
    class EpisodeRewardLogger(BaseCallback):
        def __init__(self, verbose=0):
            super(EpisodeRewardLogger, self).__init__(verbose)
            self.episode_rewards = []  # 存储每个 episode 的总奖励

        def _on_step(self) -> bool:
            """
            每一步都会调用此方法，检查当前是否为一个完整的 episode 结束
            如果是，则将 episode 的总奖励添加到列表中
            """
            if 'episode' in self.locals.get('infos', [{}])[0]:
                info = self.locals['infos'][0]
                self.episode_rewards.append(info['episode']['r'])  # 记录 episode 奖励
            return True  # 返回 True 表示继续训练


    reward_logger = EpisodeRewardLogger()

    # 开始训练模型（总共训练 200,000 步）
    model.learn(
        # total_timesteps=200_000,  # 总共训练的步数
        total_timesteps=2000,  # 总共训练的步数
        callback=[eval_callback, reward_logger],  # 注册回调函数
        tb_log_name="PPO"  # TensorBoard 日志名称
    )

    # 保存训练完成后的最终模型
    model.save("ppo_lunarlander")

    # 加载已保存的模型
    model = PPO.load("ppo_lunarlander")

    # 评估模型性能（默认使用 10 个 episode）
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    # 测试模型并渲染
    # 创建一个用于渲染的独立环境（非向量化）
    render_env = gym.make("LunarLander-v3", render_mode="human")
    # 重置环境
    obs, _ = render_env.reset()

    for _ in range(1000):  # 运行最多1000步
        action, _states = model.predict(obs, deterministic=True)  # 使用模型预测动作
        obs, reward, terminated, truncated, info = render_env.step(action)  # 执行动作
        if terminated or truncated:  # 判断 episode 是否结束
            obs, _ = render_env.reset()  # 重置环境

    # 关闭渲染窗口
    render_env.close()

    # 可视化训练过程中每个 episode 的回报曲线
    plt.figure(figsize=(12, 6))
    plt.plot(reward_logger.episode_rewards, alpha=0.6, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('PPO Training Performance on LunarLander-v3')
    plt.grid(True)

    # 添加移动平均线（平滑波动）
    window_size = 10
    moving_avg = np.convolve(reward_logger.episode_rewards,
                             np.ones(window_size) / window_size,
                             mode='valid')
    plt.plot(
        range(window_size - 1, len(reward_logger.episode_rewards)),
        moving_avg,
        'r-', linewidth=2,
        label=f'{window_size}-Episode Moving Avg'
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig("ppo_lunarlander_rewards.png", dpi=300)
    plt.show()
