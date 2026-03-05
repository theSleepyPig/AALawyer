import json
import re
import os
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ========== 模型路径和设备 ==========
model_path = "/mnt/ssd_2/yxma/LeLLM/internlm3-8b-instruct/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 加载模型和分词器
print(f"Model Path: {model_path}")

# 加载模型和分词器
# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     torch_dtype="auto",
#     device_map=device_map_setting  # 动态设备映射
# )
# tokenizer = AutoTokenizer.from_pretrained(model_path)

# 加载模型和分词器
print("正在加载模型和分词器 (InternLM3)...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map=device # ✅ 使用这个自动映射
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)
print("模型加载完成。")

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
    # 1. 构建 InternLM3 的标准 messages 格式
    system_prompt = "You are an AI assistant for legal analysis."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    # 2. 使用 apply_chat_template 生成 tokenized_chat
    tokenized_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # 3. 模型生成
    generated_ids = model.generate(
        tokenized_chat,
        max_new_tokens=2048
    )

    # 4. 截取并解码新生成的内容
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(tokenized_chat, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response.strip()

# ========== 输入输出路径 ==========
input_json = "/home/yxma/hzx/LeLLM/LawBench/data/zero_shot/4-1.json"
output_json = "/home/yxma/hzx/LeLLM/ckpt/predictions/zero_shot/4-1-150-inter-v2023.json"

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
