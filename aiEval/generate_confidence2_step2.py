import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================= 配置 =================
input_file = "results/scr_confidence_3_1_fixed.json" 
original_data_file = "/home/yxma/hzx/LeLLM/LawBench/data/zero_shot/3-1.json"
output_img = "results/risk_coverage_analysis_fixed.png"
output_csv = "results/threshold_analysis_table.csv" # 结果保存路径

# ================= 核心解析逻辑 (保持你确认过的版本) =================

def parse_law_articles(text, is_ground_truth=False):
    """
    通用法条提取函数
    """
    if not isinstance(text, str) or not text:
        return []
    
    # 1. 范围约束
    if not is_ground_truth:
        match = re.search(r"\[法条\](.*?)<eoa>", text, re.S)
        if match:
            text = match.group(1)
        else:
            pass 

    # 2. 数字提取
    nums = re.findall(r"\d+", text)
    
    # 3. 过滤与转换
    valid_nums = []
    for n in nums:
        val = int(n)
        if 1 <= val <= 1000: 
            valid_nums.append(val)
            
    return sorted(list(set(valid_nums)))

# ================= 主流程 =================

print("1. 加载原始数据以获取正确的 Ground Truth...")
gt_map = {} 
with open(original_data_file, 'r', encoding='utf-8') as f:
    gt_data = json.load(f)
    for idx, item in enumerate(gt_data):
        raw_answer = item.get('answer', '')
        gt_map[idx] = parse_law_articles(raw_answer, is_ground_truth=True)

print("2. 加载预测结果文件...")
pred_data = []
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)
except json.JSONDecodeError:
    print("警告：JSON文件可能截断，尝试读取有效部分...")
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
        try:
            fixed_content = content[:content.rfind('},')+1] + ']'
            pred_data = json.loads(fixed_content)
        except:
            print("错误：无法自动修复 JSON 文件。")
            exit()

print(f"成功加载 {len(pred_data)} 条预测记录")

# 3. 重新计算 F1
recalculated_results = []

for item in pred_data:
    idx = item['id']
    if idx not in gt_map: continue
    
    true_articles = gt_map[idx]
    raw_output = item.get('raw_output', '')
    pred_articles = parse_law_articles(raw_output, is_ground_truth=False)
    
    pred_set = set(str(x) for x in pred_articles)
    true_set = set(str(x) for x in true_articles)
    
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    recalculated_results.append({
        'confidence': item['confidence'],
        'f1_score': f1,
        'true_articles': true_articles,
        'pred_articles': pred_articles
    })

df = pd.DataFrame(recalculated_results)
baseline_f1 = df['f1_score'].mean()
print(f"\n【修正后的】Baseline F1: {baseline_f1:.4f}")

# 4. 生成统计数据 (使用更密集的采样以获得平滑曲线)
thresholds = np.linspace(0.0, 0.995, 200) # 200个点
stats = []

for t in thresholds:
    accepted = df[df['confidence'] >= t]
    if len(accepted) > 0:
        avg_f1 = accepted['f1_score'].mean()
        coverage = len(accepted) / len(df)
        num_accepted = len(accepted)
    else:
        avg_f1 = 1.0 # 或者 np.nan，视具体定义
        coverage = 0.0
        num_accepted = 0
        
    stats.append({
        'Threshold': t, 
        'Performance (F1)': avg_f1, 
        'Coverage': coverage,
        'Num_Samples': num_accepted
    })

stats_df = pd.DataFrame(stats)

# 保存完整表
stats_df.to_csv(output_csv, index=False, float_format="%.4f")
print(f"\n完整统计数据已保存至: {output_csv}")

# ================= 5. 生成 Rebuttal 专用表格 =================

print("\n" + "="*60)
print("【表1：Rebuttal 核心数据 - 按覆盖率 (Target Coverage)】")
print("说明：展示当你想保留多少样本时，系统的性能可以提升到多少。")
print("-" * 60)
print(f"{'Target Coverage':<18} | {'Actual Cov (%)':<15} | {'Threshold':<10} | {'F1 Score':<10}")
print("-" * 60)

# 关键覆盖率节点：100% -> 50%
target_coverages = [1.0, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50]

for target in target_coverages:
    # 找到覆盖率最接近 target 的那一行
    closest_idx = (stats_df['Coverage'] - target).abs().idxmin()
    row = stats_df.iloc[closest_idx]
    
    t_val = row['Threshold']
    c_val = row['Coverage'] * 100
    p_val = row['Performance (F1)']
    
    print(f"{target*100:<18.0f}% | {c_val:<15.2f} | {t_val:<10.3f} | {p_val:<10.4f}")
print("="*60)


print("\n" + "="*60)
print("【表2：Rebuttal 辅助数据 - 按置信度阈值 (Threshold)】")
print("-" * 60)
print(f"{'Threshold (τ)':<15} | {'Coverage (%)':<15} | {'F1 Score':<15}")
print("-" * 60)

# 关键阈值节点
target_thresholds = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]
for t_target in target_thresholds:
    # 找到阈值最接近 t_target 的那一行
    closest_idx = (stats_df['Threshold'] - t_target).abs().idxmin()
    row = stats_df.iloc[closest_idx]
    
    t_val = row['Threshold']
    c_val = row['Coverage'] * 100
    p_val = row['Performance (F1)']
    
    print(f"{t_val:<15.2f} | {c_val:<15.2f} | {p_val:<15.4f}")
print("="*60 + "\n")

# ================= 6. 绘图 (保持不变) =================
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.set_xlabel('Confidence Threshold (τ)', fontsize=12)
ax1.set_ylabel('System Performance (Average F1)', color='tab:red', fontsize=12)
# F1 曲线
ax1.plot(stats_df['Threshold'], stats_df['Performance (F1)'], color='tab:red', linewidth=3, label='Performance (F1)')
ax1.tick_params(axis='y', labelcolor='tab:red')
ax1.set_ylim(bottom=0.0, top=1.05) # 设定 F1 轴范围，看起来更整洁

ax2 = ax1.twinx()
ax2.set_ylabel('Retention Rate (Coverage)', color='tab:blue', fontsize=12)
# Coverage 曲线
ax2.plot(stats_df['Threshold'], stats_df['Coverage'], color='tab:blue', linestyle='--', linewidth=2, label='Coverage')
ax2.tick_params(axis='y', labelcolor='tab:blue')
ax2.set_ylim(bottom=0.0, top=1.05)

plt.title(f'Risk-Coverage Curve (Baseline F1: {baseline_f1:.4f})')
plt.grid(True, alpha=0.3)

# 图例合并
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center left')

plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"Risk-Coverage 曲线图已保存至 {output_img}")