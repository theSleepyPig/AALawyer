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

def evaluate_prediction_vs_full_law(data):
    law_db_path = "/mnt/ssd_2/yxma/LeLLM/data/data/RAGDatabase1_v2.json"
    
    with open(law_db_path, "r", encoding="utf-8") as f:
        law_data = json.load(f)

    full_law_text = "\n".join(law_data.values()).strip()

    scores = []
    for item in data:  # 因为你是 read_json() 返回的是 list
        prediction_text = item["prediction"]
        prediction_text = prediction_text.strip().split("<eoa>")[0].strip()

        if not prediction_text:
            scores.append(0.0)
            continue

        score = lcs(prediction_text, full_law_text) / max(len(prediction_text), 1)
        scores.append(score)

    average_score = sum(scores) / len(scores)
    return {
        "score": average_score,
        "count": len(scores),
        "individual_scores": scores
    }
