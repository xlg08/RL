import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.utils.data import DataLoader
import os
from tqdm import tqdm  # 引入进度条

# --- 超参数设置 ---
LEARNING_RATE = 5e-5  # 微调通常使用更小的学习率
BATCH_SIZE = 8
EPOCHS = 3
MAX_LENGTH = 256  # 增加长度以容纳人设和对话
MODEL_NAME = "gpt2"
SAVE_PATH = "models/customer_service_sft"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_and_tokenize_data(tokenizer):
    """
    加载并处理CSV格式的客服对话数据：
    1. 支持CSV文件格式
    2. 按对话ID分组处理
    3. 构造客服对话格式的输入输出对
    4. 仅对回答部分计算Loss (Label Masking)
    """
    # 读取CSV文件
    dataset = load_dataset("csv", data_dir="data", data_files="customer_support_data_samples.csv")
    # 获取训练数据集
    raw_dataset = dataset['train']

    def process_function(examples):
        inputs = []
        targets = []

        # 按conv_id分组处理对话
        conv_id_list = examples['conv_id']
        role_list = examples['role']
        text_list = examples['text']

        # 使用字典存储每个对话的轮次
        conversations = {}
        for i in range(len(conv_id_list)):
            conv_id = conv_id_list[i]
            role = role_list[i]
            text = text_list[i]

            if conv_id not in conversations:
                conversations[conv_id] = []
            conversations[conv_id].append({'role': role, 'text': text})

        # 处理每个对话 - 提取第一条记录
        for conv_id, turns in conversations.items():
            if len(turns) >= 2:  # 确保至少有一轮完整的对话
                # 查找第一条customer消息和对应的agent回复
                customer_text = None
                agent_text = None

                for turn in turns:
                    if turn['role'] == 'customer' and customer_text is None:
                        customer_text = turn['text']
                    elif turn['role'] == 'agent' and customer_text is not None and agent_text is None:
                        agent_text = turn['text']
                        break  # 找到对应的agent回复就停止

                # 如果找到了完整的对话对，则添加到训练数据中
                if customer_text is not None and agent_text is not None:
                    input_text = f"Customer: {customer_text}\nAgent:"
                    target_text = f" {agent_text}"
                    inputs.append(input_text)
                    targets.append(target_text)

        # Tokenization - 使用 padding 和 truncation 确保一致性
        model_inputs = tokenizer(
            [p + t + tokenizer.eos_token for p, t in zip(inputs, targets)],
            max_length=MAX_LENGTH,
            truncation=True,  # 截断, 当输入文本的长度超过了设定的max_length时, 切掉多余的部分
            padding="max_length", # 当输入文本的长度不足设定的max_length时,在后面补上特殊的填充符号
            return_tensors="pt"
        )

        # input_ids是输入文本中每个 token 在词汇表中的唯一编号
        # labels，相当于监督学习中的目标变量y
        labels = model_inputs["input_ids"].clone()
		# 在attention_mask中,被标记为Mask(值为0)的token只有一种:填充符号(Padding Tokens)
        attention_mask = model_inputs["attention_mask"]

        # 对每个样本单独处理 prompt 长度
        for i in range(len(inputs)):
            # Mask 掉 Prompt 部分 (提问不参与计算 loss)
            prompt_ids = tokenizer.encode(inputs[i], add_special_tokens=False)
            prompt_len = len(prompt_ids)
            # 将 prompt 部分的 labels 设为 -100
            if prompt_len < MAX_LENGTH:
                labels[i, :prompt_len] = -100
            # Mask 掉 Padding 部分
            padding_mask = attention_mask[i] == 0
            labels[i, padding_mask] = -100

        model_inputs["labels"] = labels
        return model_inputs

    # 移除原始列，只保留模型需要的列
    tokenized_train = raw_dataset.map(
        process_function,
        batched=True,
        remove_columns=['conv_id', 'role', 'text']
    )

    # 划分训练集和验证集
    split_dataset = tokenized_train.train_test_split(test_size=0.1)
    train_ds = split_dataset['train']
    val_ds = split_dataset['test']

    return train_ds, val_ds


def train():
    # 1. 模型与分词器加载
    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # GPT2 没有 pad token，指定为 eos token
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

    # 2. 数据准备
    print("Processing datasets...")
    train_ds, val_ds = prepare_and_tokenize_data(tokenizer)

    # 3. 使用 DataCollator 进行动态 Padding, 使得在组成Batch的那一瞬间, 只补齐到"当前这个Batch中最长句子的长度"
    # mlm=False 表示是 Causal Language Modeling (自回归)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        collate_fn=data_collator
    )

    # 4. 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 5. 训练循环
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0

        # 使用 tqdm 显示进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for batch in progress_bar:
            # 移动数据到设备
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            # 前向传播
            # 包含 input_ids 和 labels
            outputs = model(**batch)
            # 模型自动计算的损失值
            loss = outputs.loss

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1} - Avg Train Loss: {avg_train_loss:.4f}")

        # 验证循环
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(**batch)
                total_val_loss += outputs.loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch + 1} - Validation Loss: {avg_val_loss:.4f}")
        print("-" * 50)

    # 6. 保存模型
    print(f"Saving model to {SAVE_PATH}")
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)

    # 7.简单的推理测试
    print("Running inference test...")
    test_input = "Hello, my SSO is not working as expected."
    prompt = f"Customer: {test_input}\nAgent:"

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    output_ids = model.generate(
        **inputs,  # tokenizer处理后的输入张量
        max_new_tokens=50, # 限制模型最多生成 50 个新 token
        pad_token_id=tokenizer.eos_token_id, # 指定填充 token 的 ID 为结束符 token ID
        do_sample=True, # 启用采样生成模式，引入随机性，使生成结果更加多样化
        top_p=0.9 # 启用 nucleus sampling，只从累积概率达到 90% 的词汇中采样
    )
    print(f"Input: {prompt}")
    print(f"Generated: {tokenizer.decode(output_ids[0], skip_special_tokens=True)}")


if __name__ == "__main__":
    train()
