import torch
from torch.distributions import Categorical

# 定义每个类别的概率
probs = torch.tensor([0.1, 0.3, 0.6])  # 3个类别
dist = Categorical(probs=probs)
print("dict：", dist)

# 采样
sample = dist.sample()  # 例如：tensor(2)，表示采样到第3类（索引从0开始）
print("采样结果:", sample)

# 计算概率（log_prob）
print("采样到类别2的概率:", dist.log_prob(torch.tensor(2)).exp())  # exp(log_prob) = prob