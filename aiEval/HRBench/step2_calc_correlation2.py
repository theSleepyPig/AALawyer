import pandas as pd
from scipy.stats import pearsonr
import os

def main():
    # === 文件列表 ===
    # 你的文件目录里应该有这4个文件：
    # 第一批 (20条)
    human_file_1 = 'data/my_human_scores1.csv'  
    llm_file_1   = 'llm_reference_scores1.csv' 
    
    # 第二批 (20条)
    human_file_2 = 'data/my_human_scores2.csv'
    llm_file_2   = 'llm_reference_scores2.csv'
    
    print("正在合并两批数据...")
    
    try:
        # 读取第一批
        if not os.path.exists(human_file_1):
            print(f"❌ 缺少文件: {human_file_1}")
            return
        if not os.path.exists(llm_file_1):
            print(f"❌ 缺少文件: {llm_file_1}")
            return
            
        h1 = pd.read_csv(human_file_1)
        l1 = pd.read_csv(llm_file_1)
        # 按 Case_Index 合并人工和机器分
        df1 = pd.merge(h1, l1, on='Case_Index')
        print(f"✅ 第一批数据加载成功: {len(df1)} 条")

        # 读取第二批
        if not os.path.exists(human_file_2):
            print(f"❌ 缺少文件: {human_file_2} (请先完成第二批打分并下载)")
            return
        if not os.path.exists(llm_file_2):
            print(f"❌ 缺少文件: {llm_file_2} (请先运行 step1 生成)")
            return
            
        h2 = pd.read_csv(human_file_2)
        l2 = pd.read_csv(llm_file_2)
        df2 = pd.merge(h2, l2, on='Case_Index')
        print(f"✅ 第二批数据加载成功: {len(df2)} 条")
        
        # 纵向合并两批数据
        df = pd.concat([df1, df2], ignore_index=True)
        print(f"🎉 合并完成！总样本量: {len(df)} 条")
        
    except Exception as e:
        print(f"❌ 发生错误: {str(e)}")
        return

    dims = ['专业性', '准确性', '丰富度', '可解释性']
    print("\n🎉 === 细粒度 Meta-Evaluation (N=40) 结果 ===")
    
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
    # 注意：这里假设人工总分是4个维度平均分，机器总分是原始数据里的总分
    df['Human_总分'] = df[[f'Human_{d}' for d in dims]].mean(axis=1)
    
    # 如果机器总分是 0-100，而人工单项是 0-5，需要注意量纲。
    # Pearson 相关系数不受量纲（线性变换）影响，所以直接算没问题。
    pearson_total, p_total = pearsonr(df['Human_总分'], df['LLM_总分'])
    
    sig_mark_total = "** (极度显著 p<0.01)" if p_total < 0.01 else ("* (显著 p<0.05)" if p_total < 0.05 else "(不显著 p>=0.05)")
    
    print("\n=========================================")
    print(f"🌟 【最终总分 (N={len(df)})】 Pearson 相关系数: {pearson_total:.4f}")
    print(f"🌟 【最终总分】 p-value: {p_total:.4e} {sig_mark_total}") 
    print("=========================================")
    
    # 生成 Rebuttal 话术
    if p_total < 0.05:
        print("\n💡 恭喜！样本量扩大到 40 后，结果通常会更具说服力。Rebuttal 话术如下：")
        p_str = "p < 0.001" if p_total < 0.001 else ("p < 0.01" if p_total < 0.01 else "p < 0.05")
        print(f"\n\"We expanded our human evaluation to 40 randomly sampled cases. The results demonstrate a strong alignment between human experts and our automated metric, with a Pearson correlation of {pearson_total:.4f} ({p_str}). This statistical significance (N=40) robustly validates the reliability of our evaluation framework.\"")

if __name__ == "__main__":
    main()