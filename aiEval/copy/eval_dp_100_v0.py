import requests
import json
import time
import re
import csv
from tqdm import tqdm

import certifi, os
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


# 设置 DeepSeek API
DEEPSEEK_API_KEY = "sk-0d3e0ebfafd347a2ac4dd2c12d913bce"  #⚠️
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME = "deepseek-chat"  # "deepseek-coder","deepseek-chat",deepseek-reasoner

def format_prompt(case_text, answer_text, law_numbers=None, include_law=True):
    law_part = ""
    if include_law and law_numbers:
        law_part = f"案例涉及法条：{'、'.join(law_numbers)}\n"

    return f"""你是一个资深刑法专家，擅长评估法律文书质量。
请根据提供的【案例背景】和【分析内容】，从以下三个维度为【分析内容】打分（0~5分）：
- 专业性：是否使用准确的法律术语和刑法逻辑？
- 准确性：是否正确判断了适用的法律条文？
- 丰富度：是否补充了充分的分析、引用了类似案例或法条解释？

请你严格依据案例背景作出判断，不要随意生成内容。

案例背景：
{case_text}
{law_part}
分析内容：
{answer_text}

请按如下格式输出：
专业性：x分
准确性：x分
丰富度：x分
总分：x分
"""

# 调用 DeepSeek API
def get_score(prompt):
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "你是一个严谨的刑法学者，负责对法律分析内容进行评分。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }

    try:
        response = requests.post(DEEPSEEK_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print("API Error:", e)
        return None

# 提取评分结果
def extract_scores(text):
    try:
        return {
            '专业性': float(re.search(r"专业性[:：]\s*(\d+(?:\.\d+)?)", text).group(1)),
            '准确性': float(re.search(r"准确性[:：]\s*(\d+(?:\.\d+)?)", text).group(1)),
            '丰富度': float(re.search(r"丰富度[:：]\s*(\d+(?:\.\d+)?)", text).group(1)),
            '总分': float(re.search(r"总分[:：]\s*(\d+(?:\.\d+)?)", text).group(1))
        }
    except Exception as e:
        print("❗解析评分失败:", e)
        return None

# 主函数：读取 JSON，评估并保存 CSV
def evaluate_to_csv(input_json_path, output_csv_path, analysis_key="response_analysis_with_case"):
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["index", "专业性", "准确性", "丰富度", "总分"])  # CSV 表头

        for idx, item in enumerate(tqdm(data, desc="Evaluating with DeepSeek")):
            case_text = item["input"]
            answer_text = item.get(analysis_key, "")
            law_numbers = item.get("law_numbers", [])

            prompt = format_prompt(case_text, answer_text, law_numbers)
            score_text = get_score(prompt)
            scores = extract_scores(score_text)

            if scores:
                writer.writerow([
                    idx + 1,
                    scores["专业性"],
                    scores["准确性"],
                    scores["丰富度"],
                    scores["总分"]
                ])

            time.sleep(1.5)

    print(f"✅ 评估完成，结果已保存至：{output_csv_path}")

# main
evaluate_to_csv("results/generated_10.json", "results/eval_scores_10.csv")
