# 📚 营销话术智能助手项目
本项目是一个基于强化学习（DQN）和大语言模型（Qwen）的对话系统，用于动态调整营销策略。以下是完整的使用流程说明。
## 🧩 一、项目模块结构概览
```angular2html
marketing_dialogue_rl/
├── environment/               # 强化学习环境
│   └── dialogue_env.py        
├── agents/                    # DQN Agent 实现
│   └── dqn_agent.py           # dqn智能体
│   └── offline_dqn.py         # 离线DQN模型
├── llm/                       # 大模型接口
│   └── qwen_api.py            # 阿里云 Qwen 接口
├── utils/                     # 工具函数
│   ├── state_utils.py         # 构建对话状态向量
│   ├── sentence_encoder.py    # 对输入文本进行编码，返回句向量
│   └── data_logger.py         # 记录训练数据
├── models/                    # 存放训练好的模型
│   └── dqn_marketing_model.zip # 保存训练好的dqn模型
│   └── paraphrase-multilingual-MiniLM-L12-v2  # 预训练的Embedding模型
├── data/                      # 数据文件
│   └── dialogue_logs.json      
├── train.py                   # 模型训练脚本
└── ui/                        # Streamlit 前端界面
    └── app.py                 

```
## 🛠️ 二、环境准备与依赖安装
```Bash
pip install streamlit gym stable-baselines3 scikit-learn sentence-transformers dashscope transformers torch accelerate

```
## 🚀 三、使用流程详解
### ✅ 步骤一：部署或调用大模型
使用阿里云 DashScope API（推荐）
1、注册 阿里云
2、获取 API Key
3、替换 llm/qwen_api.py 中的密钥，或者在电脑的环境变量中设置秘钥

```Bash
dashscope.api_key = "YOUR_API_KEY_HERE"
```
### ✅ 步骤二：启动 UI 对话界面

运行 Streamlit 应用：
```Bash
streamlit run ui/app.py
```
你将看到一个简易对话界面，支持以下操作：
用户输入问题
系统根据 DQN 模型选择策略并生成回复
用户点击反馈按钮
反馈信息会自动记录到 data/dialogue_logs.json

### ✅ 步骤三：收集训练数据
在 UI 界面中进行多轮对话后，dialogue_logs.json文件将被自动生成，并记录以下信息：

| 字段名     | 类型       | 说明              |
| ---------- | ---------- |-----------------|
| state      | np.ndarray | 当前状态向量（如 384 维） |
| action     | int        | Agent 选择的动作编号   |
| reward     | float      | 用户反馈带来的奖励值      |
| next_state | np.ndarray | 执行动作后的下一个状态     |
| done       | bool       | 是否结束本轮对话        |

### ✅ 步骤四：训练 DQN 模型
运行训练脚本：
```Bash
python train.py
```
该脚本将从 data/dialogue_logs.json加载数据，并训练一个 DQN策略模型，最终保存为：
models/dqn_marketing_model.zip

### ✅ 步骤五：加载训练好的模型进行推理
重启 Streamlit 应用时，如果检测到 models/dqn_marketing_model.zip 存在，则会自动加载模型并使用最新策略进行决策。

## 📚 四、项目流程

| 目标         | 方法                            |
| ------------ |-------------------------------|
| 构建对话状态 | 使用 Embedding 编码               |
| 强化学习决策 | 使用 dqn 算法训练策略网络               |
| 数据采集     | 在 UI 中记录用户反馈                  |
| 模型训练     | 从 dialogue_logs.json 加载数据进行训练 |
| 模型部署     | 将模型保存并加载用于新对话                 |
