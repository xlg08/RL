import json
import os
from typing import Dict, Any, List

project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(project_root, "data", "dialogue_logs.json")


def ensure_data_file():
    """确保数据文件存在，若不存在则创建空文件"""
    if not os.path.exists(os.path.dirname(DATA_PATH)):
        os.makedirs(os.path.dirname(DATA_PATH))
    if not os.path.exists(DATA_PATH):
        with open(DATA_PATH, 'w', encoding='utf-8') as f:
            json.dump([], f)


def load_dialogue_logs() -> List[Dict[str, Any]]:
    """加载现有交互数据"""
    ensure_data_file()
    try:
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, FileNotFoundError, IOError):
        print("⚠️ 交互文件损坏或为空，将创建新文件")
        return []


def save_dialogue_logs(data: List[Dict[str, Any]]):
    """保存数据到 JSON 文件"""
    with open(DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def log_transition(state: List[float], action: int, reward: float, next_state: List[float], done: bool):
    """
    记录一次交互过程（state -> action -> reward -> next_state）

    参数:
        state (List[float]): 当前状态（建议转换为 list）
        action (int): 动作编号
        reward (float): 奖励值
        next_state (List[float]): 下一状态
        done (bool): 是否结束对话
    """
    transition = {
        "state": state,
        "action": action,
        "reward": reward,
        "next_state": next_state,
        "done": done
    }

    data = load_dialogue_logs()
    data.append(transition)
    save_dialogue_logs(data)
    print(f"✅ 已记录一条交互数据：action={action}, reward={reward}, done={done}")
