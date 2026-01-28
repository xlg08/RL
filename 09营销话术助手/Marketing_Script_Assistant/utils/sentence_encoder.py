import os
from sentence_transformers import SentenceTransformer
import numpy as np


class SentenceEncoder:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        self.model_path = os.path.join(project_root, "models", model_name)
        self.model = SentenceTransformer(self.model_path)

    def encode(self, text: str) -> np.ndarray:
        """
        对输入文本进行编码，返回句向量
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

