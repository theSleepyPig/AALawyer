import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

# 配置 matplotlib 支持中文显示，避免图表乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS'] 
plt.rcParams['axes.unicode_minus'] = False

def load_articles(file_path):
    """从 JSONL 或 JSON 文件中提取所有的法条编号 (relevant_articles)"""
    articles = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # 兼容 JSONL (每行一个 JSON) 或 普通 JSON 列表格式
        try:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    articles.extend(item.get('meta', {}).get('relevant_articles', []))
        except json.JSONDecodeError:
            f.seek(0)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    # 逐行提取 meta 字段下的 relevant_articles
                    if 'meta' in item and 'relevant_articles' in item['meta']:
                        articles.extend(item['meta']['relevant_articles'])
                except json.JSONDecodeError:
                    continue
    return articles

def main():
    # 1. 配置文件路径 (需与你本地文件名完全一致)
    file_full = 'data_valid.json'            # 包含 17131 条的完整验证集
    file_200 = 'aieval_dataset_200.json'     # 包含 200 条的子数据集
    
    print(f"正在加载并解析数据集，这可能需要几秒钟...")
    articles_full = load_articles(file_full)
    articles_200 = load_articles(file_200)
    
    print(f"加载完成！完整集共提取到 {len(articles_full)} 个法条标签，子集共提取到 {len(articles_200)} 个法条标签。")
    
    # 2. 统计各个法条编号出现的绝对频次
    counter_full = Counter(articles_full)
    counter_200 = Counter(articles_200)
    
    # 获取两个数据集中出现过的所有法条编号，并按数字从小到大排序
    all_articles = sorted(list(set(counter_full.keys()) | set(counter_200.keys())))
    
    # 构建频次向量
    vec_full = np.array([counter_full[art] for art in all_articles])
    vec_200 = np.array([counter_200[art] for art in all_articles])
    
    # 3. 数据归一化 (计算出现频率)
    # 因为 17131 和 200 的总数据量差距巨大，必须转化为频率才能公平对比分布
    dist_full = vec_full / vec_full.sum() if vec_full.sum() > 0 else vec_full
    dist_200 = vec_200 / vec_200.sum() if vec_200.sum() > 0 else vec_200
    
    # 4. 计算相似度指标
    cos_sim = 1 - cosine(dist_full, dist_200)
    # pearson_corr, _ = pearsonr(dist_full, dist_200)
    pearson_corr, p_value = pearsonr(dist_full, dist_200)
    
    print("\n--- 📊 分布与相似度计算结果 ---")
    print(f"涉及的法条类别总数 (Union): {len(all_articles)}")
    print(f"余弦相似度 (Cosine Similarity): {cos_sim:.4f}")
    print(f"皮尔逊相关系数 (Pearson Correlation): {pearson_corr:.4f}")
    
    # 【新增这里】：打印 P 值
    print(f"显著性水平 (P-value): {p_value:.4e}")  # 使用科学计数法打印，因为通常会非常非常小
    
    # 自动判断显著性并打印学术结论
    if p_value < 0.01:
        print("✅ 结论: P-value < 0.01，说明两个数据集的分布具有极显著的高度相关性，非偶然产生！")
    elif p_value < 0.05:
        print("✅ 结论: P-value < 0.05，说明两个数据集的分布具有显著相关性。")
    else:
        print("⚠️ 结论: P-value >= 0.05，相关性在统计学上不显著（可能是由于类别数太少或随机误差）。")
    
    # print("\n--- 📊 分布与相似度计算结果 ---")
    # print(f"涉及的法条类别总数 (Union): {len(all_articles)}")
    # print(f"余弦相似度 (Cosine Similarity): {cos_sim:.4f}")
    # print(f"皮尔逊相关系数 (Pearson Correlation): {pearson_corr:.4f}")
    
    # 5. 绘制对比柱状图
    x = np.arange(len(all_articles))
    width = 0.35  # 柱子宽度
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # 绘制归一化后的频率对比
    ax.bar(x - width/2, dist_full, width, label='Full Set (17131)', alpha=0.8, color='#1f77b4')
    ax.bar(x + width/2, dist_200, width, label='Eval Set (200)', alpha=0.8, color='#d62728')
    
    ax.set_xlabel('Article Number', fontsize=15)
    ax.set_ylabel('Normalized Frequency', fontsize=14)
    # ax.set_title('整体验证集 (17,131) 与 评测子集 (200) 法条分布对比', fontsize=14, fontweight='bold')
    
    ax.set_xticks(x)
    # 根据法条数量调整标签显示方式：由于法条可能很多，使用较小字体并旋转
    ax.set_xticklabels(all_articles, rotation=80, fontsize=6)
    
    ax.legend(fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 为了防止 x 轴标签被截断，调整 layout
    plt.tight_layout()
    
    # 6. 保存高清图片以备论文或 rebuttal PDF 使用
    output_img = 'article_distribution_17131_vs_200.png'
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"\n✅ 对比图已保存为: {output_img}")
    
    plt.show()

if __name__ == "__main__":
    main()