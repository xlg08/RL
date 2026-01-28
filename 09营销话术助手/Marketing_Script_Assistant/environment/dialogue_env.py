# -*- coding: utf-8 -*-
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MarketingDialogueEnv(gym.Env):
    """
    营销对话环境类，继承自gym.Env
    用于训练智能客服代理选择合适的营销策略进行回复
    """

    def __init__(self, mode='offline'):
        """
        初始化营销对话环境
        :param mode: 运行模式，默认为'offline'离线模式
        """
        super(MarketingDialogueEnv, self).__init__()
        self.mode = mode  # 默认离线模式，使用人工反馈收集的数据进行强化学习训练
        # 定义10种营销策略动作，每个动作包含策略描述和示例
        self.actions = {
            0: ('本次回复使用：情感共鸣策略回答问题,目的是触发客户情感需求（家庭、信任、归属感）',
                '例如：“这款安全座椅，像妈妈的手一样守护宝宝每一次出行。”'),
            1: ('本次回复使用：价值重构策略回答问题,目的是将价格转化为长期投资，弱化成本感',
                '例如：“这不仅是软件，是帮您每月节省7500元成本的工具。”'),
            2: ('本次回复使用：社会认同策略回答问题,目的是利用从众心理和权威背书',
                '例如：“您行业的XX公司都在用，上周刚续约。”'),
            3: ('本次回复使用：故事叙述策略回答问题,目的是用真实案例引发共情，替代说教',
                '例如：“一位妈妈用这款净水器后，孩子再没闹过肚子。”'),
            4: ('本次回复使用：权威专业策略回答问题,目的是引用认证/数据提升可信度',
                '例如：“产品通过欧盟安全认证，测试报告显示故障率低于0.1%。”'),
            5: ('本次回复使用：紧迫感策略回答问题,目的是制造稀缺性促立即行动',
                '例如：“本周签约可免费升级，仅剩3个名额。”'),
            6: ('本次回复使用：差异化优势策略回答问题,目的是对比竞品突出独特价值',
                '例如：“同类产品只保修1年，我们提供3年+24小时响应。”'),
            7: ('本次回复使用：免费/试用策略回答问题,目的是降低决策门槛，转移风险',
                '例如：“先试用7天，无效全额退款。”'),
            8: ('本次回复使用：个性化定制策略回答问题,目的是针对需求提供专属方案',
                '例如：“根据贵公司物流需求，我们设计了分仓备货方案。”'),
            9: ('本次回复使用：异议转化策略回答问题,目的是将反对点转化为卖点',
                '例如：“您说价格高，正因我们用了航天级材料，寿命延长5倍。”')
        }
        # action_space 表示智能体（Agent）在环境中可以采取的所有动作的集合。
        # 本项目中定义了10个营销策略，每个策略对应一个动作。
        self.action_space = spaces.Discrete(len(self.actions))
        # observation_space 描述了 Agent 所能"看到"的环境状态（即输入给模型的状态）。
        # 在本项目中，状态是通过对话历史中的所有文本（包括用户和机器人的发言）进行编码并聚合得到的。
        # 词嵌入模型是paraphrase-multilingual-MiniLM-L12-v2，维度是384
        # 因此，使用的是 Box 空间类型，表示连续值状态向量，最小值-np.inf，最大值为-np.inf，长度为384。
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(384,), dtype=np.float32)  # 示例：简单状态表示

        self.state = None
        self._reset_state()

    def _reset_state(self):
        """
        重置环境状态为随机向量
        """
        self.state = np.random.rand(384).astype(np.float32)

    def reset(self, seed=None, options=None):
        """
        重置环境到初始状态
        :param seed: 随机种子
        :param options: 其他选项参数
        :return: 初始状态和额外信息字典
        """
        super().reset(seed=seed)
        return np.zeros(384, dtype=np.float32), {}

    def step(self, action):
        """
        执行一个动作并返回环境的新状态
        :param action: 要执行的动作索引
        :return: 新状态、奖励、是否结束、是否截断、额外信息
        """
        # 使用已有数据做 offline RL，不会进入step函数
        if self.mode == "offline":
            # 本案例是离线训练
            # raise NotImplementedError("离线训练模式下不应调用 step()")
            # 返回 dummy 数据，防止中断训练, 分别代表下一个状态, 奖励, terminated (是否自然结束), truncated (是否因外部限制异常结束), info(额外信息)
            return np.zeros(384, dtype=np.float32), 0.0, True, False, {}
