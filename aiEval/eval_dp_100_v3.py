# def format_prompt(case_text, answer_text, law_numbers=None, include_law=True):
#     law_part = ""
#     if include_law and law_numbers:
#         law_part = f"【参考法条编号】：{'、'.join(law_numbers)}\n"
#
#     return f"""你是一位严谨的刑法专家，请从以下三个维度对下文的分析内容进行评分（0~5分），每项评分标准如下：
#
# 1. 专业性：分析内容是否符合规范的法律术语和刑法分析逻辑，能为法律从业者提供正确参考。
# 2. 准确性：法条引用的准确性 = 法条编号匹配的准确度 × 对应法条内容描述的真实性。该指标只与两个因素有关，一是是否和参考法条编号一致，二是法条内容是否完整、真实、可靠。与分析内容质量无关。（因为数据标注原因，参考法条以外的法条也可能是正确法条，可自行判断一下，但必须包含参考法条）
# 3. 丰富度：分析是否结构清晰、内容完整，是否结合了案例具体事实与法条条文进行深入解释，是否包含补充信息或合理推理。是否能为法律从业者提供丰富的参考。
#
# 请你仅依据“案例背景”与“分析内容”进行评分，不生成无关内容。
#
# {law_part}案例背景：
# {case_text}
#
# 分析内容：
# {answer_text}
#
# 请按如下格式输出（只输出分数）：
# 专业性：x分
# 准确性：x分
# 丰富度：x分
# """

# # 四度，更改准确性描述

from openai import OpenAI
import json
import re
import time
import csv
from tqdm import tqdm


client = OpenAI(
    api_key="sk-0d3e0ebfafd347a2ac4dd2c12d913bce",  
    base_url="https://api.deepseek.com"
)


def format_prompt(case_text, answer_text, law_numbers=None, include_law=True):
    law_part = ""
    if include_law and law_numbers:
        law_part = f"【参考法条编号】：{'、'.join(law_numbers)}\n"

    return f"""你是一位严谨的刑法专家，请从以下四个维度对下文的分析内容进行评分（0~5分），每项评分标准如下：

1. 专业性：【分析内容】是否符合规范的法律术语和刑法分析逻辑，能为法律从业者提供正确分析性参考。
2. 准确性：法条引用的准确性 = 法条编号预测的准确度（0-5） × 对应法条内容描述的真实性（0-5）÷ 5 。该指标只与两个因素有关，一是是否和参考法条编号一致，二是法条内容是否完整、真实。与分析内容质量无关。
【参考法条编号】没有的都要进行减分，可以参考F1score进行计算（因为数据标注原因，【参考法条编号】以外的法条也可能是正确法条，可自行判断一下，但【参考法条编号】是最重要的，如果没有【参考法条编号】评分为0，如果缺失进行F1score）
如果【分析内容】中有完整的相关法条，代表真实性分数为5，只有法条编号预测的准确度影响准确性分数。
3. 丰富度：【分析内容】中的案例分析是否结构清晰、内容完整，是否结合了案例具体事实与法条条文进行深入解释，是否包含补充信息或合理推理。是否能为法律从业者提供丰富的参考。
4. 可解释性：【分析内容】的透明度。即，分析内容是否含有可靠材料支撑，如真实法条和相关判决案件等。法律从业者能否感到该分析很可靠。

请你仅依据"【参考法条编号】"与“【案例背景】”对“【分析内容】”进行评分，不生成无关内容。

{law_part}【案例背景】：
{case_text}

【分析内容】：
{answer_text}

请按如下格式输出（只输出分数）：
专业性：x分
准确性：x分
丰富度：x分
可解释性：x分
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
        score_kexplain = float(re.search(r"可解释性[:：]\s*(\d+(?:\.\d+)?)", text).group(1))
        score_total = round((score_zhuanye + score_zhunque + score_fengfu + score_kexplain) / 4, 2)
        return {
            "专业性": score_zhuanye,
            "准确性": score_zhunque,
            "丰富度": score_fengfu,
            "可解释性": score_kexplain,
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
        mode: {"专业性": 0, "准确性": 0, "丰富度": 0, "可解释性": 0, "总分": 0, "count": 0}
        for mode in modes
    }

    with (open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile):
        writer = csv.writer(csvfile)
        writer.writerow(["index", "模式", "专业性", "准确性", "丰富度", "可解释性", "总分"])

        for idx, item in enumerate(tqdm(data, desc="Evaluating 3 modes with DeepSeek SDK")):
            case_text = item["input"]
            law_numbers = sorted(set(item.get("articles", [])))

            for mode in modes:
                if mode == "全RAG":
                    content = "①相关法条："+ "\n" + item.get("law_articles", "") + "\n\n" + "②案例分析："+ "\n" + item.get("response_analysis_with_case", "") + "\n\n" + "③相似判决案例参考："+ "\n" + "\n".join(item.get("similar_cases", []))
                    # + "\n" + item.get("response_analysis_with_case", "")
                elif mode == "AC-RAG":
                    content = "①相关法条："+ "\n" + item.get("law_articles", "") + "\n\n" + "②案例分析："+ "\n" + item.get("response_aa", "")
                else:  # No-RAG
                    content = "①案例分析："+ "\n" + item.get("response_analysis_no_rag", "")

                prompt = format_prompt(case_text, content, law_numbers)
                print(prompt)
                score_text = get_score(prompt)
                scores = extract_scores(score_text)

                if scores:
                    writer.writerow([
                        idx + 1, mode,
                        scores["专业性"],
                        scores["准确性"],
                        scores["丰富度"],
                        scores["可解释性"],
                        scores["总分"]
                    ])
                    for k in scores:
                        score_sums[mode][k] += scores[k]
                    score_sums[mode]["count"] += 1
                else:
                    writer.writerow([idx + 1, mode, "-", "-", "-", "-", "-"])

                time.sleep(1.5)

        # 输出平均分
        writer.writerow([])
        writer.writerow(["模式", "平均专业性", "平均准确性", "平均丰富度", "平均可解释性", "平均总分"])
        for mode in modes:
            count = score_sums[mode]["count"]
            if count > 0:
                avg = {
                    k: round(score_sums[mode][k] / count, 2)
                    for k in ["专业性", "准确性", "丰富度", "可解释性", "总分"]
                }
                writer.writerow([mode, avg["专业性"], avg["准确性"], avg["丰富度"], avg["可解释性"], avg["总分"]])
            else:
                writer.writerow([mode, "-", "-", "-", "-", "-"])

    print(f"评估完成，结果保存至：{output_csv_path}")


evaluate_to_csv("results/generated_200_m20_v2.json", "results/eval_scores_200_4dim_v2.csv")
