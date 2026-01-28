# 在线api调用大模型
import os
import dashscope
from dashscope import Generation

# 从环境变量中获取 API_KEY
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

if dashscope.api_key is None:
    raise ValueError("环境变量 DASHSCOPE_API_KEY 未设置")


def call_qwen(prompt: str) -> str:
    response = Generation.call(
        model="qwen-turbo",
        prompt=prompt
    )
    return response.output.text


# 本地大模型调用
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# # 替换为 DeepSeek-R1-Distill-Qwen-1.5B 本地模型路径
# MODEL_PATH = "../models/DeepSeek-R1-Distill-Qwen-1_5b"
#
# # 加载 tokenizer 和模型
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
#
# # 创建文本生成 pipeline
# qwen_pipeline = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     device=0  # 使用 GPU（若无 GPU，请删除或设为 -1）
# )
#
#
# def call_qwen(prompt: str) -> str:
#     # 使用 DeepSeek-R1-Distill-Qwen-1.5B 模型生成回答
#     response = qwen_pipeline(
#         prompt,
#         max_new_tokens=150,  # 推荐方式
#         truncation=True,
#         num_return_sequences=1
#     )
#     return response[0]['generated_text']
#
#
