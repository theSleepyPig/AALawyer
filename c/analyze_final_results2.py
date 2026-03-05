import os
import glob
import re
import csv
import json
from tqdm import tqdm

# --- 配置 ---
# 这里要改成 test_baselines_law.py 里保存结果的文件夹
PREDICTIONS_DIR = "baseline_predictions_v1.5_2"
OUTPUT_CSV = 'baseline_predictions_v1.5_2/final_threshold_report.csv'
# PREDICTIONS_DIR = 'baseline_predictions2' 
# OUTPUT_CSV = 'baseline_predictions2/final_threshold_report.csv'

# def calculate_f1(data):
#     """直接集成计算逻辑，不依赖外部脚本"""
#     total_tp, total_fp, total_fn = 0, 0, 0
#     for item in data:
#         # 确保转为集合
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

def calculate_f1(data):
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
    # 查找所有包含 thresh 的 json 文件
    files = glob.glob(os.path.join(PREDICTIONS_DIR, '*thresh*.json'))
    
    if not files:
        print(f"❌ 在目录 '{PREDICTIONS_DIR}' 下没找到结果文件。请先运行 shell 脚本生成数据。")
        return

    results = []
    print(f"🔍 找到 {len(files)} 个文件，开始分析...")

    for file_path in tqdm(files):
        filename = os.path.basename(file_path)
        
        # ✅ 修复点：优化正则表达式
        # \d+      : 匹配整数部分
        # (?:\.\d+)? : 非捕获组，匹配小数部分（如果有）
        # 这样就不会把文件名后缀的 .json 的点给吃进去了
        match = re.search(r'thresh(\d+(?:\.\d+)?)', filename)
        
        if not match:
            print(f"⚠️ 无法从文件名解析阈值: {filename}，跳过")
            continue
            
        try:
            threshold = float(match.group(1))
        except ValueError:
            print(f"⚠️ 转换数字失败: {match.group(1)}，跳过")
            continue

        # 读取并计算
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            p, r, f1 = calculate_f1(data)
            
            results.append({
                'Threshold': threshold,
                'F1': f1,
                'Precision': p,
                'Recall': r,
                'File': filename
            })
        except Exception as e:
            print(f"❌ 处理文件 {filename} 出错: {e}")

    if not results:
        print("❌ 没有成功解析任何结果。")
        return

    # 排序
    results.sort(key=lambda x: x['Threshold'])
    
    # 找 F1 最高的
    best_result = max(results, key=lambda x: x['F1'])

    # 打印报告
    print("\n" + "="*40)
    print(f"🏆 最佳阈值 (Best Threshold): {best_result['Threshold']}")
    print(f"   F1-Score  : {best_result['F1']:.2f}%")
    print(f"   Precision : {best_result['Precision']:.2f}%")
    print(f"   Recall    : {best_result['Recall']:.2f}%")
    print("="*40 + "\n")

    # 保存 CSV
    headers = ['Threshold', 'F1', 'Precision', 'Recall', 'File']
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"📄 完整报表已保存至: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()