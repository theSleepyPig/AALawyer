# 三维度，更改准确性和丰富性描述
from openai import OpenAI
import json
import re
import time
import csv
from tqdm import tqdm


client = OpenAI(
    api_key="",
    base_url="https://api.deepseek.com"
)


def format_prompt(case_text, answer_text, law_numbers=None, include_law=True):
    law_part = ""
    if include_law and law_numbers:
        law_part = f"【参考法条编号】：{'、'.join(law_numbers)}\n"

    return f"""你是一位严谨的刑法专家，请从以下三个维度对下文的分析内容进行评分（0~5分），每项评分标准如下：

1. 专业性：分析内容是否符合规范的法律术语和刑法分析逻辑，能为法律从业者提供正确分析性参考。
2. 准确性：法条引用的准确性 = 法条编号匹配的准确度 × 对应法条内容描述的真实性。该指标只与两个因素有关，一是是否和参考法条编号一致，二是法条内容是否完整、真实、可靠。与分析内容质量无关。（因为数据标注原因，【参考法条编号】以外的法条也可能是正确法条，可自行判断，但【参考法条编号】是最重要的，权重最高）
3. 丰富度：分析是否结构清晰、内容完整丰富，是否结合了案例具体事实与法条条文进行深入细致的解释，是否包含补充信息或合理推理。

请你仅依据“【案例背景】”与“【分析内容】”进行评分，不生成无关内容。

{law_part}【案例背景】：
{case_text}

【分析内容】：
{answer_text}

请按如下格式输出（只输出分数）：
专业性：x分
准确性：x分
丰富度：x分
"""


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
        return response.choices[0].message.content
    except Exception as e:
        print("API Error:", e)
        return None


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
        print("解析评分失败:", e)
        return None


def evaluate_to_csv(input_json_path, output_csv_path):
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    modes = ["全RAG", "AC-RAG", "No-RAG"]
    score_sums = {
        mode: {"专业性": 0, "准确性": 0, "丰富度": 0, "总分": 0, "count": 0}
        for mode in modes
    }

    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["index", "模式", "专业性", "准确性", "丰富度", "总分"])

        for idx, item in enumerate(tqdm(data, desc="Evaluating 3 modes with DeepSeek SDK")):
            case_text = item["input"]
            law_numbers = sorted(set(item.get("articles", [])))

            for mode in modes:
                if mode == "全RAG":
                    content = "相关法条：\n" + item.get("law_articles", "") + "\n案例分析：\n" + item.get("response_analysis_with_case", "")
                elif mode == "AC-RAG":
                    content = "相关法条：\n" + item.get("law_articles", "") + "\n案例分析：\n" + item.get("response_aa", "")
                else:
                    content = "案例分析：\n" + item.get("response_analysis_no_rag", "")

                prompt = format_prompt(case_text, content, law_numbers)
                score_text = get_score(prompt)
                scores = extract_scores(score_text)

                if scores:
                    writer.writerow([
                        idx + 1, mode,
                        scores["专业性"],
                        scores["准确性"],
                        scores["丰富度"],
                        scores["总分"]
                    ])
                    for k in scores:
                        score_sums[mode][k] += scores[k]
                    score_sums[mode]["count"] += 1
                else:
                    writer.writerow([idx + 1, mode, "-", "-", "-", "-"])

                time.sleep(1.5)

        # 输出平均分
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

    print(f"评估完成，结果保存至：{output_csv_path}")


evaluate_to_csv("results/generated_10.json", "results/eval_scores_10_no_explain.csv")
