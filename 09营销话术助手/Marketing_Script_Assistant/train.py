from environment.dialogue_env import MarketingDialogueEnv  # 导入营销对话环境
from agents.dqn_agent import DQNAgent  # 导入DQN智能体
import json


def load_training_data(data_path="./data/dialogue_logs.json"):
    """
    加载训练数据

    Args:
        data_path (str): 训练数据文件路径，默认为"./data/dialogue_logs.json"

    Returns:
        list: 包含对话日志的列表，如果加载失败则返回空列表
    """
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ 加载训练数据共 {len(data)} 条")
        return data
    except Exception as e:
        print(f"❌ 加载数据失败：{e}")
        return []


def run_training():
    """
    运行训练流程
    初始化营销对话环境和DQN智能体，加载训练数据并开始训练
    """
    # 创建营销对话环境实例
    env = MarketingDialogueEnv()
    # 创建DQN智能体实例
    agent = DQNAgent(env)

    # 加载训练数据
    training_data = load_training_data()

    # 检查是否有训练数据
    if len(training_data) == 0:
        print(" 无训练数据，请先运行UI界面收集对话数据")
        return

    print(" 开始训练……")
    # 开始训练，训练步数为数据条数的10倍
    agent.train(total_timesteps=len(training_data) * 10, dataset=training_data)
    # 保存训练好的模型
    agent.save()


# 程序入口点
if __name__ == "__main__":
    run_training()
