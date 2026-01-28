"""
数据源：customer_support_data_samples.csv客服数据，仅提取 Customer 的提问作为 Prompt。
模型加载：
    Actor (策略模型)：SFT微调后的模型 customer_service_sft。
    Critic (价值模型)：在 SFT 模型基础上加一个 Value Head。
    Ref Model (参考模型)：加载 customer_service_sft 并冻结，用于计算 KL 散度。
    Reward Model (奖励模型)：使用 AutoModelForSequenceClassification 加载奖励模型customer_service_rm。
奖励计算：调用 RM 模型进行预测。
"""
import torch
from torch import nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm

# --- 超参数设置 ---
# 路径配置
SFT_MODEL_PATH = "models/customer_service_sft"
RM_MODEL_PATH = "models/customer_service_rm"
SAVE_PATH = "models/customer_service_ppo"

# 训练参数 (已针对防崩塌优化)
LEARNING_RATE = 1e-6  # PPO通常需要极小的学习率
BATCH_SIZE = 4  # 这里的 Batch Size 是指 Prompt 的数量，显存敏感，因为要加载4个模型，所以设置的较小
PPO_EPOCHS = 1  # 遍历数据集次数
UPDATE_EPOCHS = 2  # 每次采集数据（Rollout）后，使用这些数据更新多少次参数
MIN_RESPONSE_LENGTH = 15
MAX_RESPONSE_LENGTH = 50  # 生成回复的最大长度
CLIP_RANGE = 0.2  # PPO Clip范围
BETA = 0.2  # KL 惩罚系数
ENTROPY_COEF = 0.01  # 熵系数，防止复读机
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ValueHead(nn.Module):
    """
    价值头：将Transformer的隐藏层状态映射为标量Value
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.value = nn.Linear(self.hidden_size, 1)
        nn.init.normal_(self.value.weight, std=0.01)
        nn.init.zeros_(self.value.bias)

    def forward(self, hidden_states):
        return self.value(hidden_states).squeeze(-1)


class PPOModel(nn.Module):
    """
    Actor-Critic 联合模型
    Actor: 生成文本 (CausalLM)
    Critic: 评估状态价值 (ValueHead)
    """

    def __init__(self, model_path):
        super().__init__()
        # 加载 SFT 后的模型作为底座
        self.llm = AutoModelForCausalLM.from_pretrained(model_path)
        self.v_head = ValueHead(self.llm.config)

    def forward(self, input_ids, attention_mask):
        # 获取 transformer 输出
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True  # 需要 Hidden States 计算 Value
        )
        # Actor Logits: [Batch, Seq, Vocab]
        logits = outputs.logits
        # Critic Values: [Batch, Seq]
        # 取最后一层隐藏状态计算 Value
        last_hidden_state = outputs.hidden_states[-1]
        values = self.v_head(last_hidden_state)
        return logits, values

    def generate(self, *args, **kwargs):
        return self.llm.generate(*args, **kwargs)


class PromptDataset(Dataset):
    """
    只加载 Customer 的提问作为 Prompt
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.prompts = []
        # 加载数据
        dataset = load_dataset("csv", data_dir="data", data_files="customer_support_data_samples.csv")['train']

        # 提取 Prompt
        seen_prompts = set()
        for i in range(len(dataset)):
            if dataset['role'][i] == 'customer':
                text = dataset['text'][i]
                if text and text not in seen_prompts:
                    # 构造符合 SFT 训练时的 Prompt 格式
                    fmt_prompt = f"Customer: {text}\nAgent:"
                    self.prompts.append(fmt_prompt)
                    seen_prompts.add(text)

        print(f"Loaded {len(self.prompts)} unique prompts for PPO.")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        # 此时只返回文本，Tokenize 在 Collate 或 Loop 中做
        return self.prompts[idx]


def get_log_probs(logits, labels):
    """
    计算生成序列的 Log Probability
    """
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    # 从模型输出的完整词汇表概率分布中，提取实际标签（labels）对应的对数概率值。
    # 例：
    # log_probs (vocab_size=4):
    # [[[0.1, 0.2, 0.3, 0.4]], [[0.4, 0.3, 0.2, 0.1]]]

    # labels:
    # [[2], [0]]

    # gather 结果:
    # [[[0.3]], [[0.4]]]
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


def compute_gae(rewards, values, masks, gamma=0.99, lam=0.95):
    """
    计算广义优势估计 (GAE) - PPO 的核心数学部分
    """
    # 将价值函数值乘以掩码，忽略无效位置的值
    values = values * masks
    # 初始化优势函数数组，与奖励张量形状相同
    advs = torch.zeros_like(rewards).to(DEVICE)
    # 初始化最后一步的GAE值为0
    last_gae = 0
    # 获取序列长度
    seq_len = rewards.shape[1]
    # 逆序遍历时间步，从最后一步开始计算GAE
    for t in reversed(range(seq_len)):
        # 如果是最后一步，则下一个状态的价值为0
        if t == seq_len - 1:
            next_value = 0
        # 否则使用下一步的价值估计
        else:
            next_value = values[:, t + 1]
        # 计算TD误差: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        # 计算GAE: A_t = δ_t + γ*λ*A_{t+1}
        last_gae = delta + gamma * lam * last_gae
        # 将计算得到的优势值存储，并应用掩码
        advs[:, t] = last_gae * masks[:, t]
    # 计算回报值: R_t = A_t + V(s_t)
    returns = advs + values
    # 返回优势函数值和回报值
    return advs, returns


def train_ppo():
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    # 生成任务使用左填充,[Pad, Pad, Prompt]
    # 模型看到的最后一个词是 Prompt 的结尾，于是能正常续写。
    tokenizer.padding_side = "left"

    # 1. 正在训练的模型 (Actor + Critic)，策略模型 + 价值头
    model = PPOModel(SFT_MODEL_PATH).to(DEVICE)

    # 2. 参考模型 (Reference Model) - 用于计算 KL 散度，防止模型跑偏
    # 为了节省显存，可以加载 float16 或量化版本
    ref_model = AutoModelForCausalLM.from_pretrained(SFT_MODEL_PATH).to(DEVICE)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # 3. 奖励模型 (Reward Model) - 你的判分器
    # 注意：这里直接加载 AutoModelForSequenceClassification
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        RM_MODEL_PATH,
        num_labels=1
    ).to(DEVICE)
    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False
    # 数据准备
    dataset = PromptDataset(tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"Starting PPO Training (Batch={BATCH_SIZE})...")

    for epoch in range(PPO_EPOCHS):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

        for batch_prompts in progress_bar:
            # --- 步骤 1: Rollout (生成数据) ---
            # 这里的 batch_prompts 是文本列表 ["Customer: xxx\nAgent:", ...]
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(
                DEVICE)
            prompt_len = inputs['input_ids'].shape[1]

            # --- 步骤 2: Rollout (生成回复) ---
            with torch.no_grad():
                # 使用 Actor 生成回复
                model.eval()
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_RESPONSE_LENGTH,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    min_length=prompt_len + MIN_RESPONSE_LENGTH
                )
                model.train()   # 切换回训练模式准备更新

            # 构造完整的序列 (Prompt + Response)
            # outputs 包含了 input_ids + generated_ids
            full_seqs = outputs
            # 创建注意力掩码，标记非填充位置为1，填充位置为0
            # full_seqs != tokenizer.pad_token_id: 生成布尔张量，非填充token位置为True，填充位置为False
            # .long(): 将布尔值转换为整数(1表示True，0表示False)，形成标准的注意力掩码格式
            attention_mask = (full_seqs != tokenizer.pad_token_id).long()

            # --- 步骤 3: 计算奖励 (Reward + KL) ---
            with torch.no_grad():
                # 3.1 RM 打分
                # RM 接收完整的句子
                rm_outputs = reward_model(input_ids=full_seqs, attention_mask=attention_mask)
                # RM分数
                rm_scores = rm_outputs.logits.squeeze(-1)  # [BATCH]

                # 3.2 Ref LogProbs (用于计算 Reward 中的 KL 惩罚)
                ref_outputs = ref_model(input_ids=full_seqs, attention_mask=attention_mask)
                ref_logits = ref_outputs.logits

                # 3.3 # 需要再跑一次当前模型，获取 Logits 和 Values，这个old_values就是critic模型的打分，用于计算GAE
                old_logits, old_values = model(full_seqs, attention_mask)

                # 对齐 Logits 和 Labels [B, Seq-1]
                # logits[:, :-1, :]
                    # 含义：取logits张量的前sequence_length - 1个位置
                    # 维度：[batch_size, sequence_length - 1, vocab_size]
                    # 目的：获取模型对每个位置的预测分布（除了最后一个位置）
                # full_seqs[:, 1:]
                    # 含义：取 full_seqs 张量的后 sequence_length - 1 个位置
                    # 维度：[batch_size, sequence_length-1]
                    # 目的：获取目标标签序列（除了第一个位置）,这是因为因果语言模型，位置t的输入预测t+1的token,所以要错位对齐，例如：
                    #         输入序列: [token_0, token_1, token_2, ..., token_{T - 1}]
                    #         目标标签: [token_1, token_2, token_3, ..., token_T]
                # get_log_probs是获取对应的选取的动作下的概率
                old_log_probs = get_log_probs(old_logits[:, :-1, :], full_seqs[:, 1:])
                ref_log_probs = get_log_probs(ref_logits[:, :-1, :], full_seqs[:, 1:])


                # 3.4 计算 KL 散度
                kl_div = old_log_probs - ref_log_probs

                # 3.5 构造 Reward 序列
                # PPO 通常是: Reward = -Beta * KL + RM_Score(只加在最后)
                rewards = -BETA * kl_div

                # 将 RM 分数加到每个句子的结束位置
                # 注意：full_seqs 包含 padding，我们需要找到每个样本真实的最后一个 token
                for i in range(len(batch_prompts)):
                    # 寻找生成的 EOS 位置
                    gen_seq = full_seqs[i, prompt_len:]
                    eos_indices = (gen_seq == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                    if len(eos_indices) > 0:
                        end_idx = eos_indices[0].item() + prompt_len - 1
                    else:
                        end_idx = rewards.shape[1] - 1

                    # 防止越界
                    end_idx = min(end_idx, rewards.shape[1] - 1)

                    # 裁剪 RM 分数防止数值不稳定
                    # clamp()将每个元素值限制在-5到5范围内
                    score = torch.clamp(rm_scores[i], -5, 5)
                    # 将裁剪后的奖励模型分数累加到指定位置的奖励序列中
                    # rewards[i, end_idx]: 第i个样本在end_idx位置的奖励值
                    # score: 经过裁剪的奖励模型输出分数
                    rewards[i, end_idx] += score

                # 创建一个训练掩码 Train Mask，用于标识哪些位置的tokens需要参与训练计算。
                # attention_mask 的长度与 full_seqs 相同，但在计算 log_probs 时，使用的是 logits[:, :-1, :] 和 full_seqs[:, 1:] 进行对齐
                # 因此需要将 attention_mask 也进行相同的切片操作 [:, 1:] 来保持维度一致
                train_mask = attention_mask[:, 1:].clone()
                # 将提示部分 mask 掉
                train_mask[:, :prompt_len - 1] = 0

                # 3.6 计算 GAE (优势函数)
                # # values 需要切片对齐 [B, Seq-1], 并且使用 detach 确保不传梯度
                old_values = old_values[:, :-1].detach()
                advantages, returns = compute_gae(rewards, old_values, train_mask)

                # Advantage Normalization
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # --- 步骤 4: PPO 更新循环 ---

            # Detach 变量防止梯度重复计算
            old_log_probs = old_log_probs.detach()
            advantages = advantages.detach()
            returns = returns.detach()

            for _ in range(UPDATE_EPOCHS):
                # 重新前向传播 (这是 PPO On-Policy 的要求）
                new_logits, new_values = model(full_seqs, attention_mask)
                new_log_probs = get_log_probs(new_logits[:, :-1, :], full_seqs[:, 1:])
                new_values = new_values[:, :-1]

                # 计算 Ratio
                # ratio = exp(new_log - old_log)
                # 只计算 train_mask 为 1 的部分
                log_ratio = (new_log_probs - old_log_probs) * train_mask
                ratio = torch.exp(log_ratio)

                # Policy Gradient Loss (PPO Clip)
                pg_loss1 = -advantages * ratio
                pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE)
                pg_loss = torch.max(pg_loss1, pg_loss2)
                pg_loss_val = (pg_loss * train_mask).sum() / (train_mask.sum() + 1e-8)

                # Value Loss
                v_loss = (new_values - returns) ** 2
                v_loss_val = (v_loss * train_mask).sum() / (train_mask.sum() + 1e-8)

                # Entropy Loss (防止复读机)
                probs = torch.softmax(new_logits[:, :-1, :], dim=-1)
                entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
                entropy_loss = -ENTROPY_COEF * (entropy * train_mask).sum() / (train_mask.sum() + 1e-8)

                # 总 Loss
                loss = pg_loss_val + 0.5 * v_loss_val + entropy_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # 严格的梯度裁剪
                optimizer.step()

            progress_bar.set_postfix({
                'reward': f"{rm_scores.mean().item():.2f}",
                'loss': f"{loss.item():.4f}",
                'kl': f"{kl_div.mean().item():.4f}"
            })

            # 清理显存
            del inputs, outputs, full_seqs, new_logits, old_logits
            torch.cuda.empty_cache()

    print(f"Saving PPO model to {SAVE_PATH}")
    # 只保存 LLM 部分，Value Head 不需要保存用于推理
    model.llm.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)


if __name__ == "__main__":
    train_ppo()
