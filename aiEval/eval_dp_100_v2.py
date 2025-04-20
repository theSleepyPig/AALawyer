from openai import OpenAI
import json
import re
import time
import csv
from tqdm import tqdm

# ✅ 配置 DeepSeek API
client = OpenAI(
    api_key="sk-0d3e0ebfafd347a2ac4dd2c12d913bce",  #⚠️
    base_url="https://api.deepseek.com"
)

# ✅ Prompt 构造
# def format_prompt(case_text, answer_text, mode):
#     return f"""你是一个资深刑法专家，擅长评估法律文书质量。
# 请根据提供的【案例背景】和【分析内容】，从以下三个维度为【分析内容】打分（0~5分）：
# - 专业性：是否使用准确的法律术语和刑法逻辑？
# - 准确性：是否正确判断了适用的法律条文？
# - 丰富度：是否补充了充分的分析、引用了类似案例或法条解释？
#
# 请你严格依据案例背景作出判断，不要随意生成内容。
#
# 案例背景：
# {case_text}
#
# 分析内容：
# {answer_text}
#
# 请按如下格式输出：
# 专业性：x分
# 准确性：x分
# 丰富度：x分
# """

def format_prompt(case_text, answer_text, law_numbers=None, include_law=True):
    law_part = ""
    if include_law and law_numbers:
        law_part = f"【参考法条编号】：{'、'.join(law_numbers)}\n"

    return f"""你是一位严谨的刑法专家，请从以下三个维度对下文的分析内容进行评分（0~5分），每项评分标准如下：

1. 专业性：分析内容是否使用了规范的法律术语和刑法逻辑，表达是否符合刑法分析惯例。
2. 准确性：= 法条匹配的准确度 × 法条内容描述的真实性。即，是否引用了正确的法条，以及是否准确反映了该法条的法律含义。
3. 丰富度：分析是否结构清晰、内容完整，是否结合了案例具体事实与法条条文进行深入解释，是否包含补充信息或合理推理。

请你仅依据“案例背景”与“分析内容”进行评分，不生成无关内容。

{law_part}案例背景：
{case_text}

分析内容：
{answer_text}

请按如下格式输出（只输出分数）：
专业性：x分
准确性：x分
丰富度：x分
"""



# ✅ 调用 DeepSeek 模型评分
def get_score(prompt):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个严谨的刑法学者，负责对法律分析内容进行评分。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        print(response)

        return response.choices[0].message.content
    except Exception as e:
        print("❗API Error:", e)
        return None

# ✅ 提取分数
def extract_scores(text):
    try:
        score_zhuanye = float(re.search(r"专业性[:：]\s*(\d+(?:\.\d+)?)", text).group(1))
        score_zhunque = float(re.search(r"准确性[:：]\s*(\d+(?:\.\d+)?)", text).group(1))
        score_fengfu = float(re.search(r"丰富度[:：]\s*(\d+(?:\.\d+)?)", text).group(1))
        score_total = round((score_zhuanye + score_zhunque + score_fengfu) / 3, 2)
        return {
            "专业性": score_zhuanye,
            "准确性": score_zhunque,
            "丰富度": score_fengfu,
            "总分": score_total
        }
    except Exception as e:
        print("❗解析评分失败:", e)
        return None

# ✅ 主函数：三模式评估 + 平均分统计
def evaluate_to_csv(input_json_path, output_csv_path):
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    modes = ["全RAG", "AC-RAG", "No-RAG"]
    score_sums = {mode: {"专业性": 0, "准确性": 0, "丰富度": 0, "总分": 0, "count": 0} for mode in modes}

    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["index", "模式", "专业性", "准确性", "丰富度", "总分"])

        for idx, item in enumerate(tqdm(data, desc="Evaluating 3 modes with DeepSeek SDK")):
            case_text = item["input"]

            # 模式 1：全 RAG
            mode = "全RAG"
            content = item.get("law_articles", "") + "\n" + item.get("response_analysis_with_case", "")
            # law_numbers = item.get("articles", [])
            law_numbers = sorted(set(item.get("articles", [])))
            prompt = format_prompt(case_text, content, law_numbers)

            print(law_numbers)
            print(prompt)

            score_text = get_score(prompt)
            scores = extract_scores(score_text)
            if scores:
                writer.writerow([idx + 1, mode, scores["专业性"], scores["准确性"], scores["丰富度"], scores["总分"]])
                for k in scores:
                    score_sums[mode][k] += scores[k]
                score_sums[mode]["count"] += 1
            else:
                writer.writerow([idx + 1, mode, "-", "-", "-", "-"])

            time.sleep(1.5)

            # 模式 2：AC-RAG
            mode = "AC-RAG"
            content = item.get("law_articles", "") + "\n" + item.get("response_aa", "")
            law_numbers = sorted(set(item.get("articles", [])))
            prompt = format_prompt(case_text, content, law_numbers)

            score_text = get_score(prompt)
            scores = extract_scores(score_text)
            if scores:
                writer.writerow([idx + 1, mode, scores["专业性"], scores["准确性"], scores["丰富度"], scores["总分"]])
                for k in scores:
                    score_sums[mode][k] += scores[k]
                score_sums[mode]["count"] += 1
            else:
                writer.writerow([idx + 1, mode, "-", "-", "-", "-"])

            time.sleep(1.5)

            # 模式 3：No-RAG
            mode = "No-RAG"
            content = item.get("response_analysis_no_rag", "")
            law_numbers = sorted(set(item.get("articles", [])))
            prompt = format_prompt(case_text, content, law_numbers)

            score_text = get_score(prompt)
            scores = extract_scores(score_text)
            if scores:
                writer.writerow([idx + 1, mode, scores["专业性"], scores["准确性"], scores["丰富度"], scores["总分"]])
                for k in scores:
                    score_sums[mode][k] += scores[k]
                score_sums[mode]["count"] += 1
            else:
                writer.writerow([idx + 1, mode, "-", "-", "-", "-"])

            time.sleep(1.5)

        # 输出每种模式的平均分
        writer.writerow([])
        writer.writerow(["模式", "平均专业性", "平均准确性", "平均丰富度", "平均总分"])
        for mode in modes:
            count = score_sums[mode]["count"]
            if count > 0:
                avg = {
                    k: round(score_sums[mode][k] / count, 2)
                    for k in ["专业性", "准确性", "丰富度", "总分"]
                }
                writer.writerow([mode, avg["专业性"], avg["准确性"], avg["丰富度"], avg["总分"]])
            else:
                writer.writerow([mode, "-", "-", "-", "-"])

    print(f"✅ 评估完成，结果保存至：{output_csv_path}")

# ✅ 示例调用
evaluate_to_csv("results/generated_10.json", "results/eval_scores_10_modes.csv")
