from stable_baselines3 import DQN
from stable_baselines3.common.logger import Logger, configure
class OfflineDQN(DQN):
    """自定义DQN类，完全绕过环境交互"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 禁用所有与环境交互相关的设置
        self.learning_starts = 0
        # self.train_freq = (1, "step")
        self.train_freq = 1
        self.gradient_steps = 1
        self.replay_buffer = None  # 稍后我们会设置它
        # 正确初始化 Logger
        self._logger = configure()
    def _setup_learn(self, total_timesteps: int, **kwargs):
        """重写_setup_learn方法，避免检查env"""
        # 不调用父类的_setup_learn方法
        pass
    def learn(self, total_timesteps: int, **kwargs) -> "OfflineDQN":
        """完全离线学习，不与环境交互"""
        # 1. 准备训练
        self._setup_learn(total_timesteps)

        # 2. 自定义训练循环
        for step in range(total_timesteps):
            # 只从回放缓冲区学习，不与环境交互
            self.train(gradient_steps=self.gradient_steps, batch_size=self.batch_size)

            # 定期更新目标网络
            if step % self.target_update_interval == 0:
                self._on_step()

            # 日志记录（可选）
            if step % 100 == 0:
                self.logger.record("train/step", step)

        return self
