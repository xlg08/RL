# -*- coding: utf-8 -*-
import numpy as np


# 把输入的两个字符串通过 "-" 连接,便于使用上述定义的 P、R 变量
def join(str1, str2):
    return str1 + '-' + str2


def sample(MDP, Pi, timestep_max, number):
    """
    序列采样方法
    @param MDP: MDP过程
    @param Pi: 策略
    @param timestep_max: 序列上限
    @param number: 采样多少条序列
    @return: 序列采样结果
    """
    episodes = []
    S, A, P, R, gamma = MDP
    for _ in range(number):
        episode = []  # 列表存储序列
        timestep = 0
        s = S[np.random.randint(4)]  # 随机选除 s5 外状态
        while s != "s5" and timestep <= timestep_max:
            timestep += 1       # 本条序列长度+1
            # rank为[0，1]均匀分布的概率，temp为累计概率和
            rand, temp = np.random.rand(), 0

            # 根据策略选动作
            for a_opt in A:
                temp += Pi.get(join(s, a_opt), 0)
                if temp > rand:
                    a = a_opt
                    r = R.get(join(s, a), 0)
                    break

            rand, temp = np.random.rand(), 0

            # 根据转移概率选下一状态
            for s_opt in S:
                temp += P.get(join(join(s, a), s_opt), 0)
                if temp > rand:
                    s_next = s_opt
                    break
            episode.append((s, a, r, s_next))
            s = s_next
        episodes.append(episode)
    return episodes


S = ["s1", "s2", "s3", "s4", "s5"]  # 状态集合
A = ["保持 s1", "前往 s1", "前往 s2", "前往 s3", "前往 s4", "前往 s5", "概率前往"]  # 动作集合
# 状态转移函数
P = {
    "s1-保持 s1-s1": 1.0, "s1-前往 s2-s2": 1.0,
    "s2-前往 s1-s1": 1.0, "s2-前往 s3-s3": 1.0,
    "s3-前往 s4-s4": 1.0, "s3-前往 s5-s5": 1.0,
    "s4-前往 s5-s5": 1.0, "s4-概率前往-s2": 0.2,
    "s4-概率前往-s3": 0.4, "s4-概率前往-s4": 0.4,
}
# 奖励函数
R = {
    "s1-保持 s1": -1, "s1-前往 s2": 0,
    "s2-前往 s1": -1, "s2-前往 s3": -2,
    "s3-前往 s4": -2, "s3-前往 s5": 0,
    "s4-前往 s5": 10, "s4-概率前往": 1,
}
# 策略 1,随机策略
Pi_1 = {
    "s1-保持 s1": 0.5, "s1-前往 s2": 0.5,
    "s2-前往 s1": 0.5, "s2-前往 s3": 0.5,
    "s3-前往 s4": 0.5, "s3-前往 s5": 0.5,
    "s4-前往 s5": 0.5, "s4-概率前往": 0.5,
}
gamma = 0.5  # 折扣因子
MDP = (S, A, P, R, gamma)


# 对所有采样序列计算所有状态的价值
def MC(episodes, V, N, gamma):
    for episode in episodes:        # 遍历出采样的每条序列
        G = 0       # 初始化的回报
        for i in range(len(episode) - 1, -1, -1):  # 一个序列从后往前计算
            (s, a, r, s_next) = episode[i]
            G = r + gamma * G
            N[s] = N[s] + 1
            V[s] = V[s] + (G - V[s]) / N[s]


timestep_max = 20
# 采样 1000 次,可以自行修改
episodes = sample(MDP, Pi_1, timestep_max, 1000)
gamma = 0.5
V = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}           # 初始化的每个状态的状态价值
N = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}           # 初始化的每个状态的计数
MC(episodes, V, N, gamma)
print("使用蒙特卡洛方法计算 MDP 的状态价值为\n", V)
#  {'s1': -1.1908045565944971, 's2': -1.653172183905843, 's3': 0.5738514322837968, 's4': 6.154660838938014, 's5': 0}
