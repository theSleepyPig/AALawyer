import os
import glob
import re
import json
import csv

# --- 配置 ---
# PREDICTIONS_DIR = 'baseline_predictions2'  # 你的结果文件夹
# OUTPUT_CSV = 'baseline_predictions2/topk_evaluation_report.csv'
PREDICTIONS_DIR = "baseline_predictions_v1.5_2"
OUTPUT_CSV = 'baseline_predictions_v1.5_2/topk_evaluation_report.csv'

# def calculate_metrics(data):
#     total_tp, total_fp, total_fn = 0, 0, 0
#     for item in data:
#         truth = set(item['ground_truth'])
#         pred = set(item['predicted'])

#         tp = len(truth.intersection(pred))
#         fp = len(pred.difference(truth))
#         fn = len(truth.difference(pred))

#         total_tp += tp
#         total_fp += fp
#         total_fn += fn

#     precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
#     recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
#     f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
#     return precision * 100, recall * 100, f1 * 100

def calculate_metrics(data):
    """
    计算混合指标：
    - Precision: Micro (全局 TP / 全局 TP+FP)
    - Recall:    Micro (全局 TP / 全局 TP+FN)
    - F1:        Macro (先算每个样本的 F1，再求平均)
    """
    # 1. Micro P/R 的累加器
    total_tp, total_fp, total_fn = 0, 0, 0
    
    # 2. Macro F1 的列表
    f1_list = []

    for item in data:
        # 提取集合
        truth = set(item['ground_truth'])
        pred = set(item['predicted'])

        # 计算当前样本的 TP, FP, FN
        tp = len(truth.intersection(pred))
        fp = len(pred.difference(truth))
        fn = len(truth.difference(pred))

        # 累加到全局 (用于 Micro P/R)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # 计算当前样本的 F1 (用于 Macro F1)
        # 注意：这里要处理分母为0的情况
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        if (p + r) > 0:
            sample_f1 = 2 * p * r / (p + r)
        else:
            sample_f1 = 0
        
        f1_list.append(sample_f1)

    # --- 计算最终指标 ---
    
    # Micro Precision
    if (total_tp + total_fp) > 0:
        micro_precision = total_tp / (total_tp + total_fp)
    else:
        micro_precision = 0.0

    # Micro Recall
    if (total_tp + total_fn) > 0:
        micro_recall = total_tp / (total_tp + total_fn)
    else:
        micro_recall = 0.0

    # Macro F1 (平均值)
    if len(f1_list) > 0:
        macro_f1 = sum(f1_list) / len(f1_list)
    else:
        macro_f1 = 0.0

    return micro_precision * 100, micro_recall * 100, macro_f1 * 100

def main():
    # 查找所有包含 topk 的 json 文件
    files = glob.glob(os.path.join(PREDICTIONS_DIR, '*topk*.json'))
    
    results = []
    print(f"🔍 正在分析 {PREDICTIONS_DIR} ...")

    for file_path in files:
        filename = os.path.basename(file_path)
        
        # 正则提取 topk 的值 (忽略带 thresh 的文件，只看纯 topk 结果)
        # 假设文件名格式: predictions_sailer_topk1.json
        match = re.search(r'topk(\d+)\.json', filename)
        
        if not match:
            continue # 跳过带 threshold 的文件或不符合格式的文件
            
        k_value = int(match.group(1))
        
        # 只关心 k=1, 2, 3 (如果你想看更多，去掉这个 if 即可)
        if k_value not in [1, 2, 3]:
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        p, r, f1 = calculate_metrics(data)
        
        results.append({
            'Top-K': k_value,
            'F1': f1,
            'Precision': p,
            'Recall': r,
            'File': filename
        })

    # 按 K 值排序
    results.sort(key=lambda x: x['Top-K'])

    # 打印表格
    print("\n" + "="*55)
    print(f"{'Top-K':<10} | {'F1-Score':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 55)
    for res in results:
        print(f"{res['Top-K']:<10} | {res['F1']:.2f}%     | {res['Precision']:.2f}%     | {res['Recall']:.2f}%")
    print("="*55 + "\n")
    
    # 保存 CSV
    headers = ['Top-K', 'F1', 'Precision', 'Recall', 'File']
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)
    print(f"📄 结果已保存至 {OUTPUT_CSV}")

if __name__ == "__main__":
    main()