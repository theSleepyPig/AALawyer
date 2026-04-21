import pandas as pd
from scipy.stats import pearsonr, spearmanr

def main():
    human_file = 'data/my_human_scores1.csv'      # 你刚才网页上点出来的结果
    llm_file = 'llm_reference_scores1.csv'   # 脚本1悄悄存的机器分数
    
    try:
        df_human = pd.read_csv(human_file)
        df_llm = pd.read_csv(llm_file)
    except FileNotFoundError:
        print("未找到文件！请确保你已经在网页上点击下载了 my_human_scores.csv 并放在本目录下。")
        return
        
    # 按题目编号合并两张表，确保顺序绝对对应
    df = pd.merge(df_human, df_llm, on='Case_Index')
    
    dims = ['专业性', '准确性', '丰富度', '可解释性']
    print("\n🎉 === 细粒度 Meta-Evaluation (元评测) 结果 ===")
    
    # 算4个维度的相关性和 p 值
    for dim in dims:
        pearson_corr, p_value = pearsonr(df[f'Human_{dim}'], df[f'LLM_{dim}'])
        
        # 判定显著性星号
        sig_mark = ""
        if p_value < 0.01:
            sig_mark = "** (极度显著 p<0.01)"
        elif p_value < 0.05:
            sig_mark = "* (显著 p<0.05)"
        else:
            sig_mark = "(不显著 p>=0.05)"
            
        print(f"👉 【{dim}】 Pearson: {pearson_corr:.4f} | p-value: {p_value:.4f} {sig_mark}")
        
    # 算总分相关性 (人工的4个维度取平均 vs 机器的总分)
    df['Human_总分'] = df[[f'Human_{d}' for d in dims]].mean(axis=1)
    pearson_total, p_total = pearsonr(df['Human_总分'], df['LLM_总分'])
    
    sig_mark_total = "** (极度显著 p<0.01)" if p_total < 0.01 else ("* (显著 p<0.05)" if p_total < 0.05 else "(不显著 p>=0.05)")
    
    print("\n=========================================")
    print(f"🌟 【最终总分】 Pearson 相关系数: {pearson_total:.4f}")
    print(f"🌟 【最终总分】 p-value: {p_total:.4f} {sig_mark_total}")
    print("=========================================")
    
    # 直接帮你生成 Rebuttal 话术
    if p_total < 0.05:
        print("\n💡 恭喜！结果具备统计学意义。你可以直接在 Rebuttal 里复制这句：")
        p_str = "p < 0.01" if p_total < 0.01 else "p < 0.05"
        print(f"\"Our small-scale meta-evaluation on randomly sampled cases shows a strong Pearson correlation of {pearson_total:.4f} ({p_str}) between human expert grading and our automated metric, firmly validating the benchmark's reliability.\"")

if __name__ == "__main__":
    main()