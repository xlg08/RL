import pandas as pd
import numpy as np
"""
    1.数据清洗
        文本里边，后边是乱码，需要清除掉
        文本里边，部分对话混杂了印地语，需要清除掉有印地语的conv_id
    2.数据抽样
        PC电脑，显存资源有限，抽取500个对话的数据
"""
def is_mostly_english(text):
    """
    判断文本是否主要是英文。因为text中混杂了印地语和英语
    剔除包含特定印地语关键词的句子 (Main, abhi, kar, raha, hoon, hai)
    """
    if not text: return False

    # 强力过滤：直接屏蔽掉常见的 Hinglish 关键词
    hinglish_keywords = ["abhi", "kar", "rahi", "kar", "raha", "ho", "gaya", "karna", "nahi"]
    for word in hinglish_keywords:
        if word.lower() in text.lower():
            return False
    return True


def clean_customer_text(text):
    """
    清理函数：保留直到最后一个标点符号的内容，去除尾部乱码
    """
    if not text or not isinstance(text, str):
        return ""

    # 预处理：去除首尾空格
    text = text.strip()

    # 定义合法的句子结束符
    # 包含了英文的 . ? ! 和中文的 。 ？ ！
    valid_endings = ['.', '?', '!', '。', '？', '！']

    # 倒序查找最后一个标点符号的位置
    last_idx = -1
    for char in valid_endings:
        idx = text.rfind(char)
        if idx > last_idx:
            last_idx = idx

    # 如果找到了标点符号
    if last_idx != -1:
        # 截取到标点符号（包含标点本身）
        cleaned = text[:last_idx + 1]
        return cleaned.strip()

    # 如果整段话都没有标点符号（极为罕见），
    # 这种数据通常质量很差，可以选择返回原文本，或者直接丢弃（返回空字符串）
    # 这里为了保守起见，返回原文本
    return text

if __name__ == '__main__':
    # 获取数据源
    data = pd.read_csv('data/customer_support_data.csv')
    print(data.info())
    data['text'] = data['text'].apply(clean_customer_text)
    # 对data数据中的text进行is_mostly_english判断
    data['is_english'] = data['text'].apply(is_mostly_english)
    # 如果is_english是False，则对应的conv_id对应的数据都删除
    # 找出包含非英文文本的 conv_id
    non_english_conv_ids = data[data['is_english'] == False]['conv_id'].unique()
    # 从数据中过滤掉这些 conv_id 对应的所有行
    data = data[~data['conv_id'].isin(non_english_conv_ids)]

    # 从数据中随机抽取500个conv_id对应的数据
    conv_ids = np.random.choice(data['conv_id'].unique(), 500, replace=False)
    data = data[data['conv_id'].isin(conv_ids)][['conv_id', 'role', 'text']]
    print(data.shape)
    data.to_csv('./data/customer_support_data_samples.csv', index=False, )
