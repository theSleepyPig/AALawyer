import openai
import json
import time


openai.api_key = ""


def format_prompt(case_text, answer_text):
    return f"""你是一个资深刑法专家，擅长评估法律文书质量。
请根据提供的【案例背景】和【分析内容】，从以下三个维度为分析打分（0-5分）：
- 专业性：是否使用准确的法律术语和刑法逻辑？
- 准确性：是否正确判断了罪名、情节、适用的法律条文？
- 丰富度：是否补充了充分的分析、引用了类似案例或法条解释？

请你严格依据案例背景作出判断，不要随意生成内容。

案例背景：
{case_text}

分析内容：
{answer_text}

请按如下格式输出：
专业性：x分
准确性：x分
丰富度：x分
总分：x分
"""


def get_score(prompt, model="gpt-4"):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print("API Error:", e)
        return None


import re
def extract_scores(text):
    try:
        scores = {}
        scores['专业性'] = float(re.search(r"专业性[:：]\s*(\d+(?:\.\d+)?)", text).group(1))
        scores['准确性'] = float(re.search(r"准确性[:：]\s*(\d+(?:\.\d+)?)", text).group(1))
        scores['丰富度'] = float(re.search(r"丰富度[:：]\s*(\d+(?:\.\d+)?)", text).group(1))
        scores['总分'] = float(re.search(r"总分[:：]\s*(\d+(?:\.\d+)?)", text).group(1))
        return scores
    except:
        return None


def evaluate_all(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    for idx, item in enumerate(data):
        print(f"Processing case {idx + 1}...")
        prompt = format_prompt(item['case'], item['model_answer'])
        score_text = get_score(prompt)
        print(score_text)
        scores = extract_scores(score_text)
        results.append({
            'case': item['case'],
            'model_answer': item['model_answer'],
            'llm_score_raw': score_text,
            'parsed_scores': scores
        })
        time.sleep(2)  # 控制频率，避免触发限速

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"评估完成，结果保存至：{output_file}")

# evaluate_all("input_cases.json", "evaluation_results.json")
evaluate_all("input_cases.json", "evaluation_results.json")
