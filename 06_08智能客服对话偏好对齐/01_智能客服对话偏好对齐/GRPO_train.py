"""
数据源：customer_support_data_samples.csv
模型加载：
    Policy Model (策略模型)：SFT微调后的模型 (不再需要 Critic/ValueHead)。
    Ref Model (参考模型)：SFT微调后的模型，冻结参数。
    Reward Model (奖励模型)：AutoModelForSequenceClassification，用于打分。
算法变更：
    PPO -> GRPO
    核心差异：移除 Value Head，对同一 Prompt 生成多条回复 (Group)，
    使用组内归一化奖励 (Group Relative Reward) 作为 Advantage。
"""
import torch
from torch import nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import random
import os
import gc
from tqdm import tqdm

# --- 超参数设置 ---
# 路径配置
SFT_MODEL_PATH = "models/customer_service_sft"
RM_MODEL_PATH = "models/customer_service_rm"
SAVE_PATH = "models/customer_service_grpo"  # 修改保存路径

# 训练参数
LEARNING_RATE = 5e-7  # GRPO 通常需要较小学习率
BATCH_SIZE = 2  # 这里的 Batch Size 是指 Prompt 的数量
GROUP_SIZE = 4  # 关键参数：每个 Prompt 生成多少个回答 (Total Batch = BATCH_SIZE * GROUP_SIZE)
# 显存警告：实际处理的并发数是 BATCH_SIZE * GROUP_SIZE = 8
GRPO_EPOCHS = 1
UPDATE_EPOCHS = 1  # 每次采集数据后，使用这些数据更新多少次参数
MIN_RESPONSE_LENGTH = 15
MAX_RESPONSE_LENGTH = 50
CLIP_RANGE = 0.2
BETA = 0.01  # KL 散度在 Loss 中的权重
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GRPOModel(nn.Module):
    """
    GRPO 只需要策略模型 (Actor)，不需要 Value Head
    """

    def __init__(self, model_path):
        super().__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(model_path)
        # 启用梯度检查点以节省显存 (可选)
        # self.llm.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask):
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False  # 不需要 hidden states
        )
        return outputs.logits

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

        seen_prompts = set()
        for i in range(len(dataset)):
            if dataset['role'][i] == 'customer':
                text = dataset['text'][i]
                if text and text not in seen_prompts:
                    fmt_prompt = f"Customer: {text}\nAgent:"
                    self.prompts.append(fmt_prompt)
                    seen_prompts.add(text)

        print(f"Loaded {len(self.prompts)} unique prompts for GRPO.")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


def get_log_probs(logits, labels):
    """
    计算生成序列的 Log Probability
    """
    # logits: [B, Seq, Vocab]
    # labels: [B, Seq]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


# --- 主训练流程 ---

def train_grpo():
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    # 生成任务使用左填充,[Pad, Pad, Prompt]
    # 模型看到的最后一个词是 Prompt 的结尾，于是能正常续写。
    tokenizer.padding_side = "left"
    # 1. 策略模型 (Policy Model) - 只需要加载一个 LLM
    model = GRPOModel(SFT_MODEL_PATH).to(DEVICE)

    # 2. 参考模型 (Reference Model)
    ref_model = AutoModelForCausalLM.from_pretrained(SFT_MODEL_PATH).to(DEVICE)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # 3. 奖励模型 (Reward Model)
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

    print(f"Starting GRPO Training (Batch={BATCH_SIZE}, Group={GROUP_SIZE})...")

    for epoch in range(GRPO_EPOCHS):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

        for batch_prompts in progress_bar:
            # batch_prompts len = BATCH_SIZE (例如 2)

            # --- 步骤 1: Input Expansion (Group 生成) ---
            # 将每个 Prompt 重复 GROUP_SIZE 次
            # 例如: [P1, P2] -> [P1, P1, P1, P1, P2, P2, P2, P2]

            expanded_prompts = []
            for p in batch_prompts:
                expanded_prompts.extend([p] * GROUP_SIZE)

            inputs = tokenizer(expanded_prompts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(
                DEVICE)
            prompt_len = inputs['input_ids'].shape[1]

            # --- 步骤 2: Rollout (生成回复) ---
            with torch.no_grad():
                model.eval()
                # 这里会同时生成 BATCH * GROUP 个序列
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_RESPONSE_LENGTH,
                    do_sample=True,  # GRPO 必须开启采样，否则同一个 Prompt 生成的 Group 是一样的
                    temperature=0.9,  # 增加多样性
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    min_length=prompt_len + MIN_RESPONSE_LENGTH
                )
                model.train()

            full_seqs = outputs
            attention_mask = (full_seqs != tokenizer.pad_token_id).long()

            # --- 步骤 3: 奖励计算 & Group Normalization ---
            with torch.no_grad():
                # 3.1 计算原始分数
                rm_outputs = reward_model(input_ids=full_seqs, attention_mask=attention_mask)
                rm_scores = rm_outputs.logits.squeeze(-1)  # [BATCH * GROUP]
                # 3.2 计算参考模型 LogProbs (用于 Loss 中的 KL 计算)
                ref_outputs = ref_model(input_ids=full_seqs, attention_mask=attention_mask)
                ref_logits = ref_outputs.logits
                ref_log_probs = get_log_probs(ref_logits[:, :-1, :], full_seqs[:, 1:])

                # 3.3 Group Normalization (核心逻辑)
                # 将分数 reshape 为 [BATCH, GROUP]
                # 例如: [s1_1, s1_2, s1_3, s1_4, s2_1, ...]
                scores_grouped = rm_scores.view(-1, GROUP_SIZE)

                # 计算组内均值和标准差
                mean_scores = scores_grouped.mean(dim=1, keepdim=True)
                std_scores = scores_grouped.std(dim=1, keepdim=True) + 1e-8  # 防止除零

                # 归一化: (Score - Mean) / Std
                advantages_grouped = (scores_grouped - mean_scores) / std_scores

                # 展平回 [BATCH * GROUP]
                advantages = advantages_grouped.view(-1)

            # 生成 Mask (只计算生成部分)
            train_mask = attention_mask[:, 1:].clone()
            train_mask[:, :prompt_len - 1] = 0

            # --- 步骤 4: GRPO 更新循环 ---

            # 预先计算旧策略的 LogProbs (Old Policy)
            with torch.no_grad():
                old_logits = model(full_seqs, attention_mask)
                # 对齐 Logits 和 Labels [B, Seq-1]
                # logits[:, :-1, :]
                    # 含义：取logits张量的前sequence_length - 1个位置
                    # 维度：[batch_size, sequence_length - 1, vocab_size]
                    # 目的：获取模型对每个位置的预测分布（除了最后一个位置）
                # full_seqs[:, 1:]
                    # 含义：取 full_seqs 张量的后 sequence_length - 1 个位置
                    # 维度：[batch_size, sequence_length-1]
                    # 目的：获取目标标签序列（除了第一个位置）,这是因为因果语言模型，位置t的输入预测t+1的token,所以要错位多起，例如：
                    #         输入序列: [token_0, token_1, token_2, ..., token_{T - 1}]
                    #         目标标签: [token_1, token_2, token_3, ..., token_T]
                old_log_probs = get_log_probs(old_logits[:, :-1, :], full_seqs[:, 1:])

            for _ in range(UPDATE_EPOCHS):
                # 前向传播 Current Policy
                new_logits = model(full_seqs, attention_mask)
                new_log_probs = get_log_probs(new_logits[:, :-1, :], full_seqs[:, 1:])

                # 计算 Ratio
                log_ratio = (new_log_probs - old_log_probs) * train_mask
                ratio = torch.exp(log_ratio)

                # 计算 Approximate KL (用于 Loss)
                # KL = exp(log_p - log_ref) - (log_p - log_ref) - 1  (Schulman estimator)
                # 或者简单的 log_p - log_ref
                # 这里使用最简单的 per-token KL: log_p - log_ref
                token_kl = (new_log_probs - ref_log_probs) * train_mask

                # GRPO Loss 公式
                # Loss = E [ min(ratio * A, clip(ratio) * A) - Beta * KL ]
                # 注意：Advantage 需要扩展到每个 Token，虽然每个 Token 的 A 是一样的

                # 将 sequence-level advantage 扩展到 token-level
                # advantages: [Batch*Group] -> [Batch*Group, 1]
                batch_adv = advantages.unsqueeze(1)
                # Policy Gradient Loss (GRPO Clip)
                pg_loss1 = -batch_adv * ratio
                pg_loss2 = -batch_adv * torch.clamp(ratio, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE)
                pg_loss = torch.max(pg_loss1, pg_loss2)

                # 加入 KL 惩罚 (DeepSeek 方式是把 KL 放在 Loss 里，而不是 Reward 里)
                # D_KL 是正数，我们希望最小化它，所以 Loss += Beta * KL
                kl_loss = BETA * token_kl

                # 总 Loss
                loss = (pg_loss + kl_loss) * train_mask
                loss = loss.sum() / train_mask.sum()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            progress_bar.set_postfix({
                'reward_mean': rm_scores.mean().item(),
                'loss': loss.item(),
                'group_std': std_scores.mean().item()  # 监控组内差异，太小说明模型崩塌(Mode Collapse)
            })

            # 清理显存
            del inputs, outputs, full_seqs, new_logits, ref_logits
            torch.cuda.empty_cache()

    print(f"Saving GRPO model to {SAVE_PATH}")
    model.llm.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)


if __name__ == "__main__":
    train_grpo()
