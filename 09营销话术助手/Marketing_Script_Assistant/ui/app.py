# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import streamlit as st

# 手动添加项目根目录到 sys.path，以便导入其他模块
PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)
# 导入自定义模块
from environment.dialogue_env import MarketingDialogueEnv  # 自定义对话环境
from agents.dqn_agent import DQNAgent  # 强化学习智能体（DQN算法）
from utils.state_utils import build_state  # 构建状态向量工具函数
from utils.data_logger import log_transition  # 记录训练数据日志工具
from llm.qwen_api import call_qwen  # 调用大模型接口

st.set_page_config(page_title="💬 营销话术智能助手 - 基于 RLHF + Qwen", layout="wide")

# 初始化 session_state 变量
# 初始化用户在文本框中输入的问题
if "new_input" not in st.session_state:
    st.session_state.new_input = ''
# 初始化强化学习环境
if "env" not in st.session_state:
    st.session_state.env = MarketingDialogueEnv()
# 初始化智能体
if "agent" not in st.session_state:
    st.session_state.agent = DQNAgent(st.session_state.env)
# 初始化对话历史
if "dialogue_history" not in st.session_state:
    st.session_state.dialogue_history = []
# 初始化对话轮数
if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0
# 初始化是否结束对话标识，如果为True，则标识为对话结束
if "done" not in st.session_state:
    st.session_state.done = False
# 初始化当前状态
if "current_state" not in st.session_state:
    st.session_state.current_state = None
# 初始化营销策略，对应的是强化学习中的动作（action）
if "action" not in st.session_state:
    st.session_state.action = None
# 获取用户反馈
if "feed_back" not in st.session_state:
    st.session_state.feed_back = None
# 初始化用户反馈对应的奖励分数，对应的是强化学习中的奖励（reward）
if "reward" not in st.session_state:
    st.session_state.reward = None
# 初始化下一步状态
if "next_state" not in st.session_state:
    st.session_state.next_state = None
# 初始化是否提交表单标识，用来控制前端页面表单的渲染
if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False

# 定义用户反馈字典，用户不同的反馈选项，对应的奖励分数不同
user_feedback_dict = {"用户成交或明确表示签约/购买": 5.0,
                      "用户表现出明显兴趣，如主动提问、索要报价": 3.0,
                      "用户要求发送产品资料、方案、合同等": 2.5,
                      "用户对产品或服务细节进行提问": 2.0,
                      "用户提出异议（如价格贵、没需求等）但仍在沟通": 0.5,
                      "用户有购买意向但表示价格有压力": 1.0,
                      "用户未表现明确态度，如说“再看看”": 0.0,
                      "用户礼貌拒绝，如“先不考虑，谢谢”": -1.0,
                      "用户明确表示没兴趣或说“不需要”": -2.0,
                      "用户长时间不回复、已读不回、敷衍应答": -2.5,
                      "用户强烈抗拒或直接中断对话（如挂电话、拉黑）": -3.0
                      }
# 侧边栏，设置一个开启对话按钮
with st.sidebar:
    if st.button("新开启对话"):
        st.session_state.dialogue_history = []
        st.session_state.turn_count = 0
        st.session_state.done = False
        st.session_state.current_state = None
        st.session_state.action = None
        st.session_state.reward = None
        st.session_state.next_state = None

st.title("💬 营销话术智能助手 - 基于 RLHF + Qwen")

# 展示历史对话记录
for msg in st.session_state.dialogue_history:
    st.markdown(f"**👤 用户：** {msg['user']}")
    st.markdown(f"**🧠 策略：** {msg['strategy']}")
    st.markdown(f"**🤖 系统：** {msg['bot']}")
    if msg.get("feedback"):
        st.markdown(f"**👤 用户反馈：** {msg['feedback']}")
    st.markdown("---")

# 接收用户输入
st.session_state.new_input = st.text_input("请输入您的问题：", key=f"user_input_{st.session_state.turn_count}")
if st.session_state.new_input:
    # 当前的状态，由历史会话+最新输入生成的
    current_state = build_state(st.session_state.dialogue_history + [{"user": st.session_state.new_input, "bot": ""}])
    # 使用DQN模型预测，预测的动作就代表的是本案例中的营销策略（营销策略是离散的，共计十种）
    action = st.session_state.agent.predict(current_state)
    st.session_state.action = action
    # 提取营销策略对应的策略描述以及参考样例
    strategy = st.session_state.env.actions[action][0]
    example = st.session_state.env.actions[action][1]
    #  拼接提示词
    prompt = f"你是一个专业的销售人员，{strategy}，用户的问题是：{st.session_state.new_input}，{example}，注意事项：不要在回答中出现策略描述"
    # 调用大模型生成回复
    response = call_qwen(prompt)
    # response = prompt  # 测试用，直接返回拼接提示词
    # 将当前状态保存到session_state
    st.session_state.current_state = current_state
    # 将回复轮次+1
    st.session_state.turn_count += 1
    # 添加记录
    new_msg = {
        "user": st.session_state.new_input,
        "strategy": strategy,
        "bot": response
    }
    st.session_state.dialogue_history.append(new_msg)
    # 这里的会话显示，目的是方便用户看到大模型回复以后，再决定给出哪个反馈
    st.markdown(f"**👤 用户：** {new_msg['user']}")
    st.markdown(f"**🧠 策略：** {new_msg['strategy']}")
    st.markdown(f"**🤖 系统：** {new_msg['bot']}")
    st.markdown(f"**👤  用户反馈：** {''}")
    st.markdown("---")
    st.session_state.form_submitted = False

# 如果用户未提交表单，则显示表单
if not st.session_state.form_submitted:
    st.markdown("### 📢 请给出人“工反馈结果：")
    # 创建表单
    with st.form(key="feedback_form", enter_to_submit=False):
        # 单选框选项
        options = list(user_feedback_dict.keys())
        # 创建单选框
        feedback_action = st.radio(
            "请给出人工反馈结果：",
            options=options,
            index=None,  # 不设置默认值
            key=f"feedback_action_{len(st.session_state.dialogue_history)}"  # 动态 Key 避免冲突
        )
        # 确认按钮（点击后提交表单）
        st.session_state.form_submitted = st.form_submit_button("确认提交")
    # 如果已提交表单
    if st.session_state.form_submitted:
        # 获取人工反馈结果
        st.session_state.feedback = feedback_action
        # 获取反馈结果对应的奖励分数
        st.session_state.reward = user_feedback_dict.get(feedback_action)
        # 历史会话添加反馈结果
        st.session_state.dialogue_history[-1]["feedback"] = feedback_action
        # 出现明确签约/购买，或者表现出明确拒绝，则判定为结束会话。done对应强化学习中的是否结束一个episode
        st.session_state.done = feedback_action in ("用户成交或明确表示签约/购买",
                                                    "用户明确表示没兴趣或说“不需要”",
                                                    "用户长时间不回复、已读不回、敷衍应答",
                                                    "用户强烈抗拒或直接中断对话（如挂电话、拉黑）"
                                                    )

        # 构建下一个状态，包含本轮对话的历史会话生成下一个状态
        st.session_state.next_state = build_state(st.session_state.dialogue_history)
        # 确保 current_state 是一个有效的 numpy.ndarray
        if st.session_state.current_state is None:
            st.session_state.current_state = np.array([])  # 或其他合适的初始状态
        # 记录状态转移
        log_transition(
            st.session_state.current_state.tolist(),
            st.session_state.action,
            st.session_state.reward,
            st.session_state.next_state.tolist(),
            st.session_state.done,
        )
        st.rerun()
