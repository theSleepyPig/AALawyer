import json
import random
import pandas as pd
import os

def main():
    # === 1. 配置路径 (请确认你的文件路径是否正确) ===
    # 原始大数据的路径 (和 step1 一样)
    json_file = r'results\m20\generated_200_m20_v2023_run1.json'
    csv_file = r'results\m20\eval_scores_m20_run1_v2023.csv'
    
    # 第一批的结果文件 (用于排除)
    run1_ref_file = 'llm_reference_scores1.csv'
    
    # 输出文件名 (生成第二批)
    html_output = 'Human_Survey_Details2.html'
    llm_reference_output = 'llm_reference_scores2.csv'

    # === 2. 读取第一批已抽取的 ID ===
    print(f"正在读取第一批结果文件: {run1_ref_file} ...")
    if not os.path.exists(run1_ref_file):
        print(f"❌ 错误：找不到 {run1_ref_file}，请确保该文件在当前目录下。")
        return
        
    df_run1 = pd.read_csv(run1_ref_file)
    used_indices = df_run1['Case_Index'].tolist()
    print(f"✅ 第一批已包含 {len(used_indices)} 个题目。")
    print(f"   排除的 ID: {sorted(used_indices)}")

    # === 3. 抽取新的 20 题 ===
    TOTAL_RANGE = range(1, 201) # 假设总共200题
    SAMPLE_SIZE = 20
    
    # 排除已抽取的
    available_indices = [i for i in TOTAL_RANGE if i not in used_indices]
    
    if len(available_indices) < SAMPLE_SIZE:
        print(f"❌ 错误：剩余题目不足 {SAMPLE_SIZE} 个！(剩余 {len(available_indices)} 个)")
        return
        
    random.seed(42)  # 固定种子保证复现
    new_sample_indices = random.sample(available_indices, SAMPLE_SIZE)
    print(f"🎉 本次新抽取的 20 个 ID: {sorted(new_sample_indices)}")
    
    # === 4. 加载原始数据并生成网页 ===
    print("正在加载原始数据...")
    if not os.path.exists(json_file):
        print(f"❌ 错误：找不到文件 {json_file}")
        return

    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        
    df = pd.read_csv(csv_file)
    # 清洗数据，确保 index 列是数字
    df = df[pd.to_numeric(df['index'], errors='coerce').notnull()]
    df['index'] = df['index'].astype(int)
    
    model_df = df[df['模式'] == '全RAG'].set_index('index')
    
    llm_data = []
    
    # 生成 HTML 头部
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>HR-Benchmark: 第二批评测 (Run 2)</title>
        <style>
            body { font-family: 'Arial', sans-serif; margin: 40px auto; max-width: 1000px; line-height: 1.6; background-color: #f4f7f6;}
            h1 { text-align: center; font-size: 26px; color: #333; }
            .case-container { background: white; border: 1px solid #ddd; padding: 25px; margin-bottom: 25px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);}
            .case-header { font-weight: bold; font-size: 18px; margin-bottom: 15px; border-bottom: 2px solid #eee; padding-bottom: 10px; color: #d35400;} 
            .fact-text { font-size: 15px; margin-bottom: 20px; color: #444; text-align: justify; line-height: 1.8;}
            details { background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; padding: 10px 15px; margin-bottom: 20px; transition: all 0.3s ease;}
            details[open] { background: #fff; border-color: #b8daff; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
            summary { font-weight: bold; font-size: 16px; cursor: pointer; outline: none; color: #0056b3; margin: -10px -15px; padding: 10px 15px;}
            summary:hover { background: #e2e6ea; border-radius: 6px;}
            .ans-section { margin-top: 15px; padding: 12px; background: #fdfdfe; border-left: 4px solid #007bff; border-radius: 4px; border-bottom: 1px solid #eee; border-right: 1px solid #eee;}
            .ans-title { font-weight: bold; color: #0056b3; margin-bottom: 8px; font-size: 14px;}
            .ans-content { font-size: 14px; color: #333; white-space: pre-wrap; font-family: monospace;}
            .scoring-area { background: #fffdfa; padding: 15px; border-radius: 5px; border: 1px dashed #ffc107;}
            .dim-row { margin-bottom: 12px; font-size: 16px; font-weight: bold; color: #333;}
            .radio-group { display: inline-block; margin-left: 20px; font-weight: normal;}
            label { margin-right: 15px; cursor: pointer; padding: 4px 8px; border-radius: 4px;}
            label:hover { background: #f0f0f0;}
            input[type="radio"] { transform: scale(1.2); margin-right: 5px; cursor: pointer;}
            .submit-btn { display: block; width: 100%; padding: 15px; font-size: 20px; font-weight: bold; color: white; background-color: #28a745; border: none; border-radius: 8px; cursor: pointer; margin-top: 30px; margin-bottom: 50px; transition: 0.2s;}
            .submit-btn:hover { background-color: #218838; }
        </style>
    </head>
    <body>
        <h1>HR-Benchmark: 专家人工评测问卷 (第二批)</h1>
        <p style="text-align: center; color: #666;">这是新增的 20 题（已自动排除第一批的题目）。请完成后下载 csv，与第一批的数据一起使用。</p>
        <hr>
        <div id="survey-form">
    """
    
    dims = ['专业性', '准确性', '丰富度', '可解释性']
    
    def format_data(data):
        if not data: return "无"
        if isinstance(data, (dict, list)):
            return json.dumps(data, ensure_ascii=False, indent=2)
        return str(data)

    for i, idx in enumerate(new_sample_indices):
        # 兼容 json 数据结构
        json_item = json_data[idx - 1] 
        fact = json_item.get('input', json_item.get('fact', '找不到案情内容')).replace('\n', '<br>')
        
        gt_laws = format_data(json_item.get('articles', '未找到数据'))
        law_content = format_data(json_item.get('law_articles', '未找到数据'))
        pred_laws = format_data(json_item.get('law_numbers', '未找到数据'))
        
        resp_aa = format_data(json_item.get('response_aa', '未找到数据'))
        resp_no_rag = format_data(json_item.get('response_analysis_no_rag', '未找到数据'))
        sim_cases = format_data(json_item.get('similar_cases', '未找到数据'))
        
        # 存 LLM 分数
        row_data = {'Case_Index': idx}
        for dim in dims:
            row_data[f'LLM_{dim}'] = model_df.loc[idx, dim]
        row_data['LLM_总分'] = model_df.loc[idx, '总分']
        llm_data.append(row_data)
        
        html_content += f"""
        <div class="case-container" data-idx="{idx}">
            <div class="case-header">题目 #{idx} (第二批第 {i+1} 题)</div>
            <div class="fact-text"><b>【案情事实】</b><br>{fact}</div>
            
            <details>
                <summary>📑 点击展开/收起：模型作答与参考答案</summary>
                
                <div class="ans-section">
                    <div class="ans-title">1. Ground Truth (真实法条编号) [articles]</div>
                    <div class="ans-content">{gt_laws}</div>
                </div>
                
                <div class="ans-section" style="border-left-color: #6f42c1;">
                    <div class="ans-title">2. 法条具体内容 [law_articles]</div>
                    <div class="ans-content">{law_content}</div>
                </div>
                
                <div class="ans-section">
                    <div class="ans-title">3. 预测法条编号 [law_numbers]</div>
                    <div class="ans-content">{pred_laws}</div>
                </div>
                
                <div class="ans-section" style="border-left-color: #28a745;">
                    <div class="ans-title">4. RAG 模式作答 [response_aa]</div>
                    <div class="ans-content">{resp_aa}</div>
                </div>
                
                <div class="ans-section" style="border-left-color: #dc3545;">
                    <div class="ans-title">5. 无 RAG 模式作答 [response_analysis_no_rag]</div>
                    <div class="ans-content">{resp_no_rag}</div>
                </div>
                
                <div class="ans-section" style="border-left-color: #ffc107;">
                    <div class="ans-title">6. 检索到的相似案例 [similar_cases]</div>
                    <div class="ans-content">{sim_cases}</div>
                </div>
            </details>
            
            <div class="scoring-area">
        """
        
        for dim in dims:
            html_content += f'<div class="dim-row">{dim}: <span class="radio-group">'
            for score in range(6):
                html_content += f'<label><input type="radio" name="{dim}_{idx}" value="{score}"> {score}</label>'
            html_content += '</span></div>'
            
        html_content += """
            </div>
        </div>
        """
        
    html_content += """
        </div>
        <button class="submit-btn" onclick="downloadCSV()">✅ 全部打完，点击下载第二批结果</button>

        <script>
        function downloadCSV() {
            let cases = document.querySelectorAll('.case-container');
            let dims = ['专业性', '准确性', '丰富度', '可解释性'];
            let csvContent = "\\uFEFFCase_Index,Human_专业性,Human_准确性,Human_丰富度,Human_可解释性\\n";
            let allFilled = true;
            
            cases.forEach(function(caseDiv) {
                let idx = caseDiv.getAttribute('data-idx');
                let row = [idx];
                
                dims.forEach(function(dim) {
                    let checked = caseDiv.querySelector('input[name="' + dim + '_' + idx + '"]:checked');
                    if (checked) {
                        row.push(checked.value);
                    } else {
                        allFilled = false;
                    }
                });
                csvContent += row.join(",") + "\\n";
            });
            
            if (!allFilled) {
                alert("⚠️ 提示：您还有题目没有打完分，请检查是否有遗漏！");
                return;
            }
            
            let blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
            let link = document.createElement("a");
            let url = URL.createObjectURL(blob);
            link.setAttribute("href", url);
            link.setAttribute("download", "my_human_scores2.csv"); // 文件名变成 2
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            alert("🎉 下载成功！请将 'my_human_scores2.csv' 保存好，用于最终合并计算。");
        }
        </script>
    </body>
    </html>
    """
    
    # 保存 LLM 参考分和 HTML
    pd.DataFrame(llm_data).to_csv(llm_reference_output, index=False, encoding='utf-8-sig')
    with open(html_output, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"\n✅ 第二批网页生成完毕！包含 {SAMPLE_SIZE} 个新题目。")
    print(f"👉 1. 打开 【{html_output}】 进行打分，下载 【my_human_scores2.csv】")
    print(f"👉 2. 确保第一批结果 (1.csv) 和 第二批结果 (2.csv) 都在文件夹里")
    print(f"👉 3. 运行新的合并计算脚本。")

if __name__ == "__main__":
    main()