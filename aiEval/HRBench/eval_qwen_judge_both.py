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

DASHSCOPE_API_KEY = "sk-a8d24875aad7411fb43e2cce0ba71596" 

client = OpenAI(
    api_key=DASHSCOPE_API_KEY, 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

JUDGE_MODEL = "qwen-plus"

# ✅ Prompt 构造（新增可解释性维度）
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

# ✅ 调用 DeepSeek 模型评分
def get_score(prompt):
    max_retries = 3
    for i in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL, # 使用 Qwen-Max
                messages=[
                    {"role": "system", "content": "你是一个严谨的刑法学者，负责对法律分析内容进行评分。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0, # 保持0以确保复现性
                top_p=0.0001   # 尽可能贪婪解码
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e): # 限流错误
                print(f"⚠️ 触发限流，等待重试 ({i+1}/{max_retries})...")
                time.sleep(5) # 增加等待时间
            else:
                print(f"❗API Error: {e}")
                return None
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
def evaluate_to_csv(main_json_path, ccs_json_path, output_csv_path):
    print(f"🚀 开始使用 {JUDGE_MODEL} 进行评测...")
    
    # 1. 分别读取两个文件
    try:
        with open(main_json_path, "r", encoding="utf-8") as f:
            data_main = json.load(f) # 包含 No-RAG, AC-RAG, 全RAG
        with open(ccs_json_path, "r", encoding="utf-8") as f:
            data_ccs = json.load(f)  # 包含 CCS-RAG
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        return

    # 简单检查长度是否一致（可选）
    if len(data_main) != len(data_ccs):
        print(f"⚠️ 警告：两个文件数据量不一致 ({len(data_main)} vs {len(data_ccs)})，程序将按较短的列表执行。")

    modes = ["全RAG", "AC-RAG", "No-RAG", "CCS-RAG"]
    score_sums = {
        mode: {"专业性": 0, "准确性": 0, "丰富度": 0, "可解释性": 0, "总分": 0, "count": 0}
        for mode in modes
    }

    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["index", "模式", "专业性", "准确性", "丰富度", "可解释性", "总分"])

        # 2. 修改循环：使用 zip 同时遍历两个列表
        # item_main 来自主文件，item_ccs 来自CCS文件
        for idx, (item_main, item_ccs) in enumerate(tqdm(zip(data_main, data_ccs), total=min(len(data_main), len(data_ccs)), desc=f"Evaluating")):
            
            # 基础信息（假设两个文件的 input 是一样的，取主文件的即可）
            case_text = item_main["input"]
            law_numbers = sorted(set(item_main.get("articles", [])))

            for mode in modes:
                content = ""
                if mode == "全RAG":
                    # 从主文件获取
                    content = "①相关法条："+ "\n" + item_main.get("law_articles", "") + "\n\n" + "②案例分析："+ "\n" + item_main.get("response_analysis_with_case", "") + "\n\n" + "③相似判决案例参考："+ "\n" + "\n".join(item_main.get("similar_cases", []))
                
                elif mode == "AC-RAG":
                    # 从主文件获取
                    content = "①相关法条："+ "\n" + item_main.get("law_articles", "") + "\n\n" + "②案例分析："+ "\n" + item_main.get("response_aa", "")
                
                elif mode == "CCS-RAG":
                    # 【关键修改】从 item_ccs 获取 response_analysis_onlyccs
                    # 注意：如果 similar_cases 在 ccs 文件里也有，建议优先用 item_ccs 的
                    ccs_analysis = item_ccs.get("response_analysis_onlyccs", "")
                    ccs_cases = item_ccs.get("similar_cases", []) 
                    content = "①案例分析："+ "\n" + ccs_analysis + "\n\n" + "②相似判决案例参考："+ "\n" + "\n".join(ccs_cases)
                
                else:  # No-RAG
                    # 从主文件获取
                    content = "①案例分析："+ "\n" + item_main.get("response_analysis_no_rag", "")

                prompt = format_prompt(case_text, content, law_numbers)
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

# ✅ 示例调用
# evaluate_to_csv("results\generated_200_m20_v2_run2.json", "results\m20_qwen\eval_scores_m20_run2.csv")
# results\generated_200_m20_v2_run2.json
# if __name__ == "__main__":

#     parser = argparse.ArgumentParser(description="使用 DeepSeek API 评估法律分析内容，并生成评分 CSV 文件。")
#     parser.add_argument("input_path", help="包含待评估内容的输入 JSON 文件路径。")
#     parser.add_argument("output_path", help="用于保存评分结果的输出 CSV 文件路径。")
#     args = parser.parse_args()

#     evaluate_to_csv(args.input_json_path, args.output_csv_path)
if __name__ == "__main__":
    # 你的主文件（包含 No-RAG, AC-RAG 等）
    main_file = "results\generated_200_m0_v2.json" 
    
    # 你的CCS文件（包含 response_analysis_onlyccs）
    ccs_file = "results\generated_200_m0_v2_ccs.json"  # <--- 请修改为实际文件名
    
    # 输出文件
    output_file = "results/m0_qwen/generated_200_m0_v2_qwen.csv"

    import os
    if not os.path.exists(main_file) or not os.path.exists(ccs_file):
        print("❌ 错误: 找不到输入文件，请检查路径。")
    else:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 调用修改后的函数
        evaluate_to_csv(main_file, ccs_file, output_file)