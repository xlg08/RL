import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# --- 配置路径 ---
GRPO_MODEL_PATH = "models/customer_service_grpo"
PPO_MODEL_PATH = "models/customer_service_ppo"  # 请确保此路径与你保存PPO模型的路径一致
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 测试用例 ---
TEST_CASES = [
    "Hello, my SSO is not working as expected.",
    "I can’t log in. It says account locked.",
    "Your product is terrible! I want a refund now."
]


def generate_response(model, tokenizer, text, model_name="Model"):
    """
    通用的推理生成函数
    """
    prompt = f"Customer: {text}\nAgent:"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,  # 最大生成长度
            min_new_tokens=10,  # 最小生成长度
            do_sample=True,  # 开启采样
            temperature=0.7,  # 控制创造性 (PPO通常需要稍微低一点的温度以保持稳定)
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,  # 惩罚重复，防止复读机
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # --- 后处理截断 ---
    # 提取 Agent 回复部分
    try:
        # 截取 "Agent:" 之后的内容
        response = generated_text.split("Agent:")[-1].strip()
        # 再次截断，防止模型自己生成下一轮的 "Customer:" 或换行
        if "Customer:" in response:
            response = response.split("Customer:")[0].strip()
        if "\n" in response:
            response = response.split("\n")[0].strip()
    except IndexError:
        response = generated_text

    print(f"\n[Customer]: {text}")
    print(f"[Agent ({model_name})]: {response}")
    print("-" * 30)


def valid_grpo():
    """
    加载并测试 GRPO 模型
    """
    print(f"\n>>> Loading GRPO model from {GRPO_MODEL_PATH}...")
    if not os.path.exists(GRPO_MODEL_PATH):
        print(f"错误：路径 {GRPO_MODEL_PATH} 不存在。")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(GRPO_MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(GRPO_MODEL_PATH).to(DEVICE)
        model.eval()

        print("=" * 50)
        print("GRPO Model Inference Test")
        print("=" * 50)

        for text in TEST_CASES:
            generate_response(model, tokenizer, text, model_name="GRPO")

        # 释放显存
        del model, tokenizer
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"加载 GRPO 模型出错: {e}")


def valid_ppo():
    """
    加载并测试 PPO 模型
    """
    print(f"\n>>> Loading PPO model from {PPO_MODEL_PATH}...")
    if not os.path.exists(PPO_MODEL_PATH):
        print(f"错误：路径 {PPO_MODEL_PATH} 不存在。请检查保存路径。")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(PPO_MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(PPO_MODEL_PATH).to(DEVICE)
        model.eval()

        print("=" * 50)
        print("PPO Model Inference Test")
        print("=" * 50)

        for text in TEST_CASES:
            generate_response(model, tokenizer, text, model_name="PPO")

        # 释放显存
        del model, tokenizer
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"加载 PPO 模型出错: {e}")


if __name__ == "__main__":
    # valid_ppo()
    valid_grpo()