# -*- coding: utf-8 -*-

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
import argparse

# ✅ 配置 DeepSeek API
client = OpenAI(
    api_key="sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",  # 替换为你的 API Key
    base_url="https://api.deepseek.com"
)

# ✅ Prompt 构造（新增可解释性维度）
def format_prompt(case_text, answer_text, law_numbers=None, include_law=True):
    law_part = ""
    if include_law and law_numbers:
        law_part = f"【参考法条编号】：{'、'.join(law_numbers)}\n"

    return f"""你是一位严谨的刑法专家，请从以下四个维度对下文的分析内容进行评分（0~5分），每项评分标准如下：

1. 专业性：【分析内容】是否符合规范的法律术语和刑法分析逻辑，能为法律从业者提供正确分析性参考。

2. 准确性： 法条引用的准确性 = 法条编号预测的准确度（0-5） × 对应法条内容描述的真实性（0-5） ÷ 5 。
法条编号预测的准确度（0-5分）评分标准：
本项仅考核“条”编号的匹配程度，不考核“款”。如果【分析内容】中引用的法条编号（条）完全覆盖了【参考法条编号】，则此项得5分。如果【分析内容】引用了【参考法条编号】以外的条，或未能引用【参考法条编号】中的条，则参考F1score进行计算（精确率和召回率的调和平均数）。【参考法条编号】是最重要的，若其中任何一条未被覆盖，此项得分将显著降低；若【参考法条编号】全部缺失，则此项为0分。因为数据标注原因，【参考法条编号】以外的法条也可能是正确法条，可自行判断一下
特别说明： 如果【分析内容】引用的条编号与【参考法条编号】一致，即使额外引用了该条款下的其他款项或其他罪名定义，也不扣分。例如，【参考法条编号】为“343条”，【分析内容】引用了“343条”下的非法采矿罪和破坏性采矿罪两款，仍视为条编号预测完全准确，此项得5分。
法条内容描述的真实性（0-5分）评分标准：
考核所引用的法条内容的真实性。只要【分析内容】中针对所引用的法条编号，提供了任何真实的、非编造的法条内容（无论是整条、部分款项或罪名定义），此项即得5分。不考核内容的完整性和引用格式。
该指标只与上述两个因素有关，与分析内容的质量无关。
如果【分析内容】中提供了任何真实的法条内容，代表真实性分数为5。
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
        return response.choices[0].message.content
    except Exception as e:
        print("❗API Error:", e)
        return None

# ✅ 提取四维评分
def extract_scores(text):
    try:
        score_zhuanye = float(re.search(r"专业性[:：]\s*(\d+(?:\.\d+)?)", text).group(1))
        score_zhunque = float(re.search(r"准确性[:：]\s*(\d+(?:\.\d+)?)", text).group(1))
        score_fengfu = float(re.search(r"丰富度[:：]\s*(\d+(?:\.\d+)?)", text).group(1))
        score_kexplain = float(re.search(r"可解释性[:：]\s*(\d+(?:\.\d+)?)", text).group(1))
        score_total = round((score_zhuanye + score_zhunque + score_fengfu + score_kexplain) / 4, 2)
        
        score_zhuanye_100 = score_zhuanye * 20
        score_zhunque_100 = score_zhunque * 20
        score_fengfu_100 = score_fengfu * 20
        score_kexplain_100 = score_kexplain * 20
        score_total = round((score_zhuanye_100 + score_zhunque_100 + score_fengfu_100 + score_kexplain_100) / 4, 2)
        
        return {
            "专业性": score_zhuanye_100,
            "准确性": score_zhunque_100,
            "丰富度": score_fengfu_100,
            "可解释性": score_kexplain_100,
            "总分": score_total
        }
    except Exception as e:
        print("❗解析评分失败:", e)
        return None

# ✅ 三种模式 + 平均分统计
def evaluate_to_csv(input_json_path, output_csv_path):
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    modes = ["全RAG", "AC-RAG", "No-RAG", "CCS-RAG"]
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
                elif mode == "CCS-RAG":
                    content = "①案例分析："+ "\n" + item.get("response_analysis_onlyccs", "") + "\n\n" + "②相似判决案例参考："+ "\n" + "\n".join(item.get("similar_cases", []))
                else:  # No-RAG
                    content = "①案例分析："+ "\n" + item.get("response_analysis_no_rag", "")

                prompt = format_prompt(case_text, content, law_numbers)
                # print
                # print(prompt)
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

    print(f"✅ 评估完成，结果保存至：{output_csv_path}")

# # ✅ 原基础代码
# # evaluate_to_csv("input_file", "result_path")
# evaluate_to_csv("results\m0\generated_200_m0_v2023_run3.json", "results\m0\eval_scores_m0_run3_v2023.csv")


# ✅ 优化代码
# if __name__ == "__main__":

#     parser = argparse.ArgumentParser(description="使用 DeepSeek API 评估法律分析内容，并生成评分 CSV 文件。")
#     parser.add_argument("input_path", help="包含待评估内容的输入 JSON 文件路径。")
#     parser.add_argument("output_path", help="用于保存评分结果的输出 CSV 文件路径。")
#     args = parser.parse_args()

#     evaluate_to_csv(args.input_json_path, args.output_csv_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用DeepSeekAPI评估法律分析内容，并生成评分CSV文件。")
    
    parser.add_argument("--input_path", type=str, required=True, help="待评估内容的输入JSON文件路径。")
    parser.add_argument("--output_path", type=str, required=True, help="保存评分结果的输出CSV文件路径。")
    
    args = parser.parse_args()

    evaluate_to_csv(args.input_path, args.output_path)