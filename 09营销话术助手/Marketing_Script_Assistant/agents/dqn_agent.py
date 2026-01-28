"""
本项目是RLHF任务，通过人工反馈进行学习。为了更接近实际使用场景，选择off-policy来进行训练。
算法层面，选用DQN算法， 它是一种off-policy算法，使用 replay buffer 存储历史经验，
支持手动扩展 replay buffer，通过 replay_buffer.add()支持外部数据，
"""

# from agents.offline_dqn import OfflineDQN
from offline_dqn import OfflineDQN
from stable_baselines3.common.buffers import ReplayBuffer
import os
import numpy as np

project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(project_root, "models", "dqn_marketing_model")


class DQNAgent:
    def __init__(self, env):
        self.env = env
        if os.path.exists(f"{MODEL_PATH}.zip"):
            self.model = OfflineDQN.load(MODEL_PATH)
            print("✅ 加载已有模型")
        else:
            self.model = OfflineDQN("MlpPolicy",
                                    self.env,
                                    verbose=1,
                                    buffer_size=100000,
                                    exploration_fraction=0.3,  # 增加探索时间比例
                                    exploration_initial_eps=1.0,  # 初始探索率
                                    exploration_final_eps=0.01,  # 最终探索率（更小的值）
                                    learning_starts=1000,  # 增加开始学习前的步数
                                    target_update_interval=500  # 更新目标网络的频率
                                    )
            self.model.exploration_rate = 1.0
            print("使用新模型初始化，并强制开启 100% 随机探索")
            print("🆕 使用新模型初始化")

    def train(self, total_timesteps, dataset=None):
        if dataset is not None:
            # # 将 dataset 转换为 replay buffer 支持的格式
            # observations = []
            # actions = []
            # rewards = []
            # next_observations = []
            # dones = []

            buffer_size = len(dataset)
            self.model.replay_buffer = ReplayBuffer(
                buffer_size,
                self.model.observation_space,
                self.model.action_space,
                device=self.model.device,
                n_envs=1
            )

            for item in dataset:
                # 每次处理一条 transition
                obs = np.array(item["state"], dtype=np.float32)
                next_obs = np.array(item["next_state"], dtype=np.float32)
                action = np.array([item["action"]], dtype=np.int8)  # shape: (1,)
                reward = np.array(item["reward"], dtype=np.float32)
                done = np.array(item["done"], dtype=bool)

                # ✅ 一条一条地添加进 buffer
                self.model.replay_buffer.add(
                    obs=obs,
                    next_obs=next_obs,
                    action=action,
                    reward=reward,
                    done=done,
                    infos=[{}]
                )
            print(f"✅ 成功向 replay buffer 添加 {len(dataset)} 条数据")

        # 开始训练
        self.model.learn(total_timesteps=total_timesteps)

    def predict(self, observation):
        # deterministic=False 表示允许探索
        action, _states = self.model.predict(observation, deterministic=False)
        return action.item()

    def save(self):
        self.model.save(MODEL_PATH)
        print(f"💾 模型已保存至 {MODEL_PATH}.zip")
if __name__ == '__main__':
    from environment.dialogue_env import MarketingDialogueEnv
    env = MarketingDialogueEnv()
    # 创建DQN智能体实例
    agent = DQNAgent(env)
    action = agent.predict(np.zeros(384))
    print(action)

