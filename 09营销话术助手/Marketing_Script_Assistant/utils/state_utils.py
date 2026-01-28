"""
对话状态处理工具模块
"""

from typing import List, Dict
import numpy as np
from utils.sentence_encoder import SentenceEncoder

encoder = SentenceEncoder()


def build_state(dialogue_history: List[Dict[str, str]]) -> np.ndarray:
    """
    构建当前对话状态表示（使用 编码）

    参数:
        dialogue_history (List[Dict]): 对话历史列表，每个元素包含 "user" 和 "bot"

    返回:
        np.ndarray: 表示当前状态的向量（384维）
    """
    if not dialogue_history:
        return np.zeros(384)  # 根据所选嵌入模型维度调整，paraphrase-multilingual-MiniLM-L12-v2维度是384

    all_utterances = []
    for turn in dialogue_history:
        all_utterances.append(turn.get("user", ""))
        all_utterances.append(turn.get("bot", ""))

    embeddings = encoder.model.encode(all_utterances, convert_to_numpy=True)
    state = np.mean(embeddings, axis=0)  # 取平均向量作为状态

    return state.astype(np.float32)


def normalize_state(state: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(state)
    return state / norm if norm != 0 else state
