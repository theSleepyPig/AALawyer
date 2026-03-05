import json
import re

def lcs(a, b):
    """ 计算字符串 a 和 b 的最长公共子序列长度 """
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
    return dp[n][m]

def chinese_to_arabic(cn):
    cn_num = {
        '零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
        '五': 5, '六': 6, '七': 7, '八': 8, '九': 9
    }
    unit = {'十': 10, '百': 100, '千': 1000}
    result = 0
    tmp = 0
    unit_val = 1
    cn = cn.replace("零", "")  # 忽略零

    for char in reversed(cn):
        if char in unit:
            unit_val = unit[char]
            if tmp == 0:
                tmp = 1
        elif char in cn_num:
            result += cn_num[char] * unit_val
            tmp = 0
        else:
            # 非中文数字跳过
            continue
    return str(result)

def extract_article_number(text):
    # 先尝试匹配阿拉伯数字：第273条
    match = re.search(r"(第)?(\d{1,4})(条)", text)
    if match:
        return str(int(match.group(2)))

    # 再尝试匹配中文数字：第二百七十三条
    match = re.search(r"第([一二三四五六七八九十百千零]{1,10})条", text)
    if match:
        chinese = match.group(1)
        return chinese_to_arabic(chinese)

    return None

def evaluate_prediction_vs_full_law(data):
    law_db_path = "/mnt/ssd_2/yxma/LeLLM/data/data/RAGDatabase1_v2.json"

    with open(law_db_path, "r", encoding="utf-8") as f:
        law_data = json.load(f)

    law_texts = list(law_data.values())

    # def best_lcs_score(prediction):
    #     best_score = 0
    #     for law in law_texts:
    #         law = law.strip()
    #         lcs_len = lcs(prediction, law)
    #         law_len = len(law)
    #         if law_len == 0:
    #             continue
    #         ratio = lcs_len / law_len
    #         best_score = max(best_score, ratio)
    #     return best_score
    
    # def best_lcs_score(prediction):
    #     best_score = 0
    #     best_match = ""
    #     for law in law_texts:
    #         law = law.strip()
    #         lcs_len = lcs(prediction, law)
    #         law_len = len(law)
    #         if law_len == 0:
    #             continue
    #         ratio = lcs_len / law_len
    #         if ratio > best_score:
    #             best_score = ratio
    #             best_match = law
    #     print("\n====== Matching Example ======")
    #     print("[Prediction]\n", prediction)
    #     print("[Best Match Score]:", round(best_score, 4))
    #     print("[Best Match Law]\n", best_match)
    #     print("================================\n")
    #     return best_score
    
    def best_lcs_score(prediction):
        article_no = extract_article_number(prediction)
        if article_no and article_no in law_data:
            law = law_data[article_no].strip()
            lcs_len = lcs(prediction, law)
            score = lcs_len / len(law) if len(law) > 0 else 0
            print("\n====== Matching Example (By Article No.) ======")
            print("[Prediction]\n", prediction)
            print("[Article No.]", article_no)
            print("[Score]:", round(score, 4))
            print("[Matched Law]\n", law)
            print("================================\n")
            return score
        else:
            # fallback 全库匹配可选：也可以直接 return 0
            print(f"\n Warning: No matching article number or law not found for: {prediction}")
            return 0.0

    scores = []
    for item in data:
        prediction_text = item["prediction"].strip()
        if not prediction_text:
            scores.append(0.0)
            continue

        score = best_lcs_score(prediction_text)
        scores.append(score)

    avg_score = sum(scores) / len(scores)
    return {
        "score": avg_score,
        "count": len(scores),
        "auth_list": scores
    }
    
def compute_hall_score(data_dict, acc_list, auth_list):
    print(f"[Debug] acc_list: {len(acc_list)}, auth_list: {len(auth_list)}, data_dict: {len(data_dict)}")
    assert len(acc_list) == len(auth_list) == len(data_dict)

    hall_list = [1 - acc * auth for acc, auth in zip(acc_list, auth_list)]
    average_hall_score = sum(hall_list) / len(hall_list)

    return {
        "score": average_hall_score,
        "hall_list": hall_list
    }
