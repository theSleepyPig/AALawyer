import json
import re
import os
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ========== 模型路径和设备 ==========
model_path = "/mnt/ssd_2/yxma/LeLLM/Qwen3-8B/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map=device  # 动态设备映射
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# ========== 法条数据库 ==========
# with open("/mnt/ssd_2/yxma/LeLLM/data/data/RAGDatabase1_v2.json", "r", encoding="utf-8") as f:
with open("/mnt/ssd_2/yxma/LeLLM/data/data/RAGDatabase_v2023_v3.json", "r", encoding="utf-8") as f:
    law_data = json.load(f)

# ========== 工具函数 ==========
def extract_law_numbers(text):
    return re.findall(r"刑法第(\d+)条", text)

def retrieve_law_articles(law_numbers):
    articles = [law_data.get(num, "未找到相关法条") for num in law_numbers]
    return "\n".join(articles)

def generate_response(prompt):
    # 1. 构建 Qwen3 的标准 messages 格式
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # 2. 使用 apply_chat_template 生成最终输入文本
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 3. 模型生成
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2048
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # 4. Qwen3 特有的 <think> 标签解析逻辑
    #    这部分直接从官方示例复制，确保正确提取最终答案
    try:
        # 寻找 </think> 标签的 token ID (151668)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    
    # 我们只需要最终的 content，不需要 thinking_content
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
    
    return content

# ========== 输入输出路径 ==========
input_json = "/home/yxma/hzx/LeLLM/LawBench/data/zero_shot/4-1.json"
output_json = "/home/yxma/hzx/LeLLM/ckpt/predictions/zero_shot/4-1-150-qwen-v2023.json"

# ========== 加载输入数据 ==========
with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)  # 直接是 list，不需要转

results = []

for idx, item in enumerate(tqdm(data, desc="Generating law articles")):
    question = item["origin_prompt"] if "origin_prompt" in item else item["question"]

    prompt = (
        "根据下列事实和罪名给出涉及的刑法法条。"
        "只需给出刑法法条编号，请将答案填在[法条]与<eoa>之间。\n"
        "例如[法条]刑法第128条、刑法第341条<eoa>\n"
        f"{question}"
    )
    
    response = generate_response(prompt)
    law_numbers = extract_law_numbers(response)
    law_articles = retrieve_law_articles(law_numbers)

    results.append({
        "input": question,
        "law_articles": law_articles
    })

# ========== 保存结果 ==========
os.makedirs(os.path.dirname(output_json), exist_ok=True)
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"✅ Done! Results saved to: {output_json}")
