# analyze_results.py
import os
import glob
import subprocess
import re
import csv
from tqdm import tqdm

# --- 配置 ---
PREDICTIONS_DIR = '/home/yxma/hzx/LeLLM/c/baseline_predictions_rerank_thresholds'
EVAL_SCRIPT = 'evaluate_fap.py'
OUTPUT_CSV = '/home/yxma/hzx/LeLLM/c/baseline_predictions_rerank_thresholds/threshold_evaluation_report.csv'

# --- 主逻辑 ---
def main():
    # 查找所有bge threshold的预测结果文件
    # json_files = glob.glob(os.path.join(PREDICTIONS_DIR, 'predictions_bge_thresh*.json'))
    json_files = glob.glob(os.path.join(PREDICTIONS_DIR, 'predictions_bge-m3_reranker_thresh*.json'))
    # /home/yxma/hzx/LeLLM/c/baseline_predictions_rerank_thresholds/predictions_bge-m3_reranker_thresh0.78.json
    
    if not json_files:
        print(f"Error: No prediction files found in '{PREDICTIONS_DIR}'.")
        print("Please run 'run_all_thresholds.sh' first.")
        return

    all_results = []

    print(f"Found {len(json_files)} result files to analyze.")

    # 循环处理每个文件
    for json_file in tqdm(json_files, desc="Analyzing results"):
        # 1. 从文件名中提取阈值
        match = re.search(r'thresh(\d+\.\d+)\.json', os.path.basename(json_file))
        if not match:
            continue
        threshold = float(match.group(1))

        # 2. 调用 evaluate_fap.py 脚本并捕获其输出
        command = ['python', EVAL_SCRIPT, json_file]
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode != 0:
            print(f"Error evaluating {json_file}:\n{result.stderr}")
            continue
        
        output = result.stdout
        
        # 3. 从输出中用正则表达式解析分数
        precision_match = re.search(r"Precision:\s+([\d\.]+)", output)
        recall_match = re.search(r"Recall:\s+([\d\.]+)", output)
        f1_match = re.search(r"F1-Score:\s+([\d\.]+)", output)

        if precision_match and recall_match and f1_match:
            precision = float(precision_match.group(1))
            recall = float(recall_match.group(1))
            f1_score = float(f1_match.group(1))
            
            all_results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            })

    # 4. 将结果按阈值排序并写入CSV文件
    if not all_results:
        print("No results were successfully parsed.")
        return
        
    all_results.sort(key=lambda x: x['threshold'])

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['threshold', 'precision', 'recall', 'f1_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n✅ Analysis complete. Report saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()