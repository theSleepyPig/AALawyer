# analyze_all_results.py
import os
import glob
import subprocess
import re
import pandas as pd
import argparse
from tqdm import tqdm

def parse_filename(filename):
    """从文件名中解析出方法和参数"""
    basename = os.path.basename(filename)
    
    # 匹配带 topk 的文件名，例如 predictions_bge_topk1.json
    match_topk = re.search(r'predictions_(.+?)_topk(\d+)\.json', basename)
    if match_topk:
        method = match_topk.group(1).upper()
        parameter = f"Top-K={match_topk.group(2)}"
        return {'method': method, 'parameter': parameter}

    # 匹配带 thresh 的文件名，例如 predictions_bge_thresh0.75.json
    match_thresh = re.search(r'predictions_(.+?)_thresh([\d\.]+)\.json', basename)
    if match_thresh:
        method = match_thresh.group(1).upper()
        parameter = f"Threshold={match_thresh.group(2)}"
        return {'method': method, 'parameter': parameter}
        
    # 匹配不带参数的文件名，例如 predictions_bge.json
    match_base = re.search(r'predictions_(.+?)\.json', basename)
    if match_base:
        method = match_base.group(1).upper()
        parameter = "Default"  # 或者可以根据您的默认设置填写，例如 "Default Top-5"
        return {'method': method, 'parameter': parameter}

    return None

def main():
    parser = argparse.ArgumentParser(description="Analyze all JSON prediction files and generate a CSV report.")
    parser.add_argument("--input_dir", type=str, default="baseline_predictions_v1.5", help="Directory containing the JSON prediction files.")
    parser.add_argument("--eval_script", type=str, default="evaluate_fap.py", help="Path to the evaluation script.")
    parser.add_argument("--output_csv", type=str, default="baseline_predictions_v1.5/full_ablation_report_v2.csv", help="Path to the output CSV report.")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found.")
        return

    json_files = glob.glob(os.path.join(args.input_dir, '*.json'))
    
    if not json_files:
        print(f"Error: No .json files found in '{args.input_dir}'.")
        return

    all_results = []
    print(f"Found {len(json_files)} result files to analyze in '{args.input_dir}'...")

    for file_path in tqdm(json_files, desc="Analyzing files"):
        parsed_info = parse_filename(file_path)
        if not parsed_info:
            print(f"Skipping file with unknown format: {file_path}")
            continue

        command = ['python', args.eval_script, file_path]
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode != 0:
            print(f"Error evaluating {file_path}:\n{result.stderr}")
            continue
        
        output = result.stdout
        
        precision_match = re.search(r"Precision:\s+([\d\.]+)", output)
        recall_match = re.search(r"Recall:\s+([\d\.]+)", output)
        f1_match = re.search(r"F1-Score:\s+([\d\.]+)", output)

        if precision_match and recall_match and f1_match:
            precision = float(precision_match.group(1))
            recall = float(recall_match.group(1))
            f1_score = float(f1_match.group(1))
            
            all_results.append({
                'Method': parsed_info['method'],
                'Parameter': parsed_info['parameter'],
                'F1-Score': f1_score,
                'Precision': precision,
                'Recall': recall
            })

    if not all_results:
        print("Could not parse any results.")
        return
        
    # 使用 pandas 创建 DataFrame 并保存为 CSV
    df = pd.DataFrame(all_results)
    df = df.sort_values(by=['Method', 'Parameter'], ascending=[True, True]) # 排序让结果更美观
    df.to_csv(args.output_csv, index=False, float_format='%.2f')

    print(f"\n✅ Analysis complete! Report saved to: {args.output_csv}")
    print("\n--- Report Preview ---")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
    