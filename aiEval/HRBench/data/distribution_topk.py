import json
import numpy as np
from collections import Counter
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

def load_charges(file_path):
    """从 JSONL 或 JSON 文件中提取所有的罪名 (accusation/charges)"""
    charges = []
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    # 优先提取 accusation，如果没有再尝试 charges
                    acc = item.get('meta', {}).get('accusation', [])
                    if not acc:
                        acc = item.get('meta', {}).get('charges', [])
                    charges.extend(acc)
        except json.JSONDecodeError:
            f.seek(0)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    acc = item.get('meta', {}).get('accusation', [])
                    if not acc:
                        acc = item.get('meta', {}).get('charges', [])
                    charges.extend(acc)
                except json.JSONDecodeError:
                    continue
    return charges

def main():
    # 1. 配置文件路径 (确保与你本地一致)
    file_full = 'data_valid.json'            # 17131 条
    file_200 = 'aieval_dataset_200.json'     # 200 条
    
    print("正在加载并解析数据集以提取【罪名】...")
    charges_full = load_charges(file_full)
    charges_200 = load_charges(file_200)
    
    print(f"提取完成！完整集共提取到 {len(charges_full)} 个罪名标签，子集共提取到 {len(charges_200)} 个罪名标签。\n")
    
    # 2. 统计频次
    counter_full = Counter(charges_full)
    counter_200 = Counter(charges_200)
    
    # 构建全部罪名集合
    all_charges = sorted(list(set(counter_full.keys()) | set(counter_200.keys())))
    
    # 3. 统计学相似度计算 (同之前的逻辑)
    vec_full = np.array([counter_full[c] for c in all_charges])
    vec_200 = np.array([counter_200[c] for c in all_charges])
    
    dist_full = vec_full / vec_full.sum() if vec_full.sum() > 0 else vec_full
    dist_200 = vec_200 / vec_200.sum() if vec_200.sum() > 0 else vec_200
    
    cos_sim = 1 - cosine(dist_full, dist_200)
    pearson_corr, p_value = pearsonr(dist_full, dist_200)
    
    print("=== 📊 罪名分布统计学指标 ===")
    print(f"余弦相似度: {cos_sim:.4f}")
    print(f"皮尔逊相关系数: {pearson_corr:.4f} (P-value: {p_value:.2e})\n")
    
    # 4. 生成 Top-5 罪名分布对比表
    print("=== 📈 Top-5 主要罪名分布对比 (可直接用于 Rebuttal) ===")
    print(f"{'Top Charges (罪名)':<25} | {'CAIL2018 Valid (17k)':<20} | {'HR-Benchmark (200)':<20}")
    print("-" * 75)
    
    # 找出完整集里排名前 5 的罪名
    top_5_charges = [charge for charge, count in counter_full.most_common(5)]
    
    total_full = sum(counter_full.values())
    total_200 = sum(counter_200.values())
    
    for charge in top_5_charges:
        # 计算在 17k 中的百分比
        pct_full = (counter_full[charge] / total_full) * 100 if total_full > 0 else 0
        # 计算在 200 中的百分比
        pct_200 = (counter_200[charge] / total_200) * 100 if total_200 > 0 else 0
        
        # 打印表格行
        print(f"{charge:<25} | {pct_full:>17.1f}% | {pct_200:>17.1f}%")
        
    print("-" * 75)
    print("提示: 你可以直接把这个表格的内容填入 LaTeX 中！")

if __name__ == "__main__":
    main()