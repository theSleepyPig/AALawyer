import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================= 配置 =================
# json_file = "results/scr_confidence_calibration_test.json"  # 你的结果文件
json_file = "results/scr_confidence_3_1_test.json"  # 你的结果文件
output_img = "results/risk_coverage_analysis2.png"          # 输出图片文件名

# ================= 加载数据 =================
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)
df = pd.DataFrame(data)

print(f"总样本数: {len(df)}")

# ================= 分析函数 =================
def analyze_thresholds(df):
    thresholds = np.linspace(0.0, 0.99, 50)
    stats = []

    # 全集（不筛选）时的正确样本数（用于计算召回率 Recall）
    # 这里定义 "正确" 为 F1 > 0 (或者你可以定义为 F1=1.0)
    total_correct_count = len(df[df['f1_score'] > 0]) 

    for t in thresholds:
        # 1. 筛选：只保留 Confidence >= t 的样本
        accepted = df[df['confidence'] >= t]
        rejected = df[df['confidence'] < t]
        
        # 2. 如果全都过滤掉了，避免除零错误
        if len(accepted) == 0:
            precision = 1.0 # 理论上没有预测就不犯错
            coverage = 0.0
            recall = 0.0
        else:
            # Precision: 留下的样本里，平均 F1 是多少？
            precision = accepted['f1_score'].mean()
            coverage = len(accepted) / len(df)
            
            # Recall (针对正确样本): 原本对的样本，现在还剩多少？
            # 统计 accepted 里 F1 > 0 的数量
            accepted_correct_count = len(accepted[accepted['f1_score'] > 0])
            recall = accepted_correct_count / total_correct_count if total_correct_count > 0 else 0

        # 3. 统计“误杀”：明明是对的 (F1=1.0)，却被拒之门外了
        # 这种样本最可惜，是你担心的“低可靠却是对的”
        false_rejections = len(rejected[rejected['f1_score'] == 1.0])
        
        stats.append({
            'Threshold': t,
            'Precision (Avg F1)': precision,
            'Coverage (Retention)': coverage,
            'Recall (of Correct)': recall,
            'False Rejections (Count)': false_rejections,
            'Accepted Count': len(accepted)
        })

    return pd.DataFrame(stats)

# ================= 运行分析 =================
results_df = analyze_thresholds(df)

# ================= 打印关键数据点 =================
# 选几个关键阈值打印出来给 Reviewer 看
print("\n=== 关键阈值分析表 ===")
print(f"{'Threshold':<10} | {'Precision':<10} | {'Coverage':<10} | {'误杀正确样本数':<15}")
print("-" * 60)
for t in [0.5, 0.8, 0.9, 0.95]:
    # 找到最接近该阈值的行
    row = results_df.iloc[(results_df['Threshold'] - t).abs().argsort()[:1]].iloc[0]
    print(f"{row['Threshold']:<10.2f} | {row['Precision (Avg F1)']:<10.4f} | {row['Coverage (Retention)']:<10.1%} | {int(row['False Rejections (Count)']):<15}")
print("-" * 60)

# ================= 画图 =================
fig, ax1 = plt.subplots(figsize=(10, 6))

# X轴：Confidence Threshold
ax1.set_xlabel('Confidence Threshold (τ)')
ax1.set_ylabel('Metrics', color='black')

# 线1：Precision (准确率) - 应该上升
line1, = ax1.plot(results_df['Threshold'], results_df['Precision (Avg F1)'], 
         color='tab:red', linewidth=2, label='Precision (Avg F1 of Accepted)')

# 线2：Recall (召回率) - 应该下降
line2, = ax1.plot(results_df['Threshold'], results_df['Recall (of Correct)'], 
         color='tab:blue', linewidth=2, linestyle='--', label='Recall (Retention of Correct)')

# 线3：Coverage (总覆盖率) - 参考用
line3, = ax1.plot(results_df['Threshold'], results_df['Coverage (Retention)'], 
         color='gray', linewidth=1, linestyle=':', label='System Coverage')

# 添加图例
lines = [line1, line2, line3]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center left')

plt.title('Risk-Coverage Analysis: Trade-off between Precision and Recall')
plt.grid(True, alpha=0.3)
plt.savefig(output_img)
print(f"\n分析图表已保存为: {output_img}")
print("结论：请观察红线（Precision）是否随阈值上升，蓝线（Recall）是否下降。")
print("如果红线显著上升，说明 sacrifice（牺牲）一部分 Recall 是值得的。")