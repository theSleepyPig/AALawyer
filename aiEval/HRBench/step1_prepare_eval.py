import json
import random
import pandas as pd

def main():
    json_file = 'results\m20\generated_200_m20_v2023_run1.json'
    csv_file = 'results\m20\eval_scores_m20_run1_v2023.csv'

    
    html_output = 'Human_Survey_Details1.html'
    llm_reference = 'llm_reference_scores1.csv'


    
    # 抽取数量设为 20
    SAMPLE_SIZE = 20
    
    print("正在加载数据...")
    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        
    df = pd.read_csv(csv_file)
    df = df[df['index'].astype(str).str.isnumeric()]
    df['index'] = df['index'].astype(int)
    
    model_df = df[df['模式'] == '全RAG'].set_index('index')
    
    random.seed(42) 
    sample_indices = random.sample(range(1, 201), SAMPLE_SIZE)
    
    llm_data = []
    
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <style>
            body { font-family: 'Arial', sans-serif; margin: 40px auto; max-width: 1000px; line-height: 1.6; background-color: #f4f7f6;}
            h1 { text-align: center; font-size: 26px; color: #333; }
            .case-container { background: white; border: 1px solid #ddd; padding: 25px; margin-bottom: 25px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);}
            .case-header { font-weight: bold; font-size: 18px; margin-bottom: 15px; border-bottom: 2px solid #eee; padding-bottom: 10px; color: #2c3e50;}
            .fact-text { font-size: 15px; margin-bottom: 20px; color: #444; text-align: justify; line-height: 1.8;}
            
            /* 折叠框样式 */
            details { background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; padding: 10px 15px; margin-bottom: 20px; transition: all 0.3s ease;}
            details[open] { background: #fff; border-color: #b8daff; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
            summary { font-weight: bold; font-size: 16px; cursor: pointer; outline: none; color: #0056b3; margin: -10px -15px; padding: 10px 15px;}
            summary:hover { background: #e2e6ea; border-radius: 6px;}
            
            .ans-section { margin-top: 15px; padding: 12px; background: #fdfdfe; border-left: 4px solid #007bff; border-radius: 4px; border-bottom: 1px solid #eee; border-right: 1px solid #eee;}
            .ans-title { font-weight: bold; color: #0056b3; margin-bottom: 8px; font-size: 14px;}
            .ans-content { font-size: 14px; color: #333; white-space: pre-wrap; font-family: monospace;}
            
            /* 打分区域样式 */
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
        <h1>HR-Benchmark: 专家人工评测问卷</h1>
        <p style="text-align: center; color: #666;">请点击<b>“查看模型作答与参考答案”</b>展开详细信息，结合法条具体内容进行阅卷，直接点击 0~5 的评分。拉到最下方保存。</p>
        <hr>
        <div id="survey-form">
    """
    
    dims = ['专业性', '准确性', '丰富度', '可解释性']
    
    def format_data(data):
        if not data: return "无"
        if isinstance(data, (dict, list)):
            return json.dumps(data, ensure_ascii=False, indent=2)
        return str(data)

    for i, idx in enumerate(sample_indices):
        json_item = json_data[idx - 1] 
        fact = json_item.get('input', json_item.get('fact', '找不到案情内容')).replace('\n', '<br>')
        
        # 提取你需要的 6 个关键字段
        gt_laws = format_data(json_item.get('articles', '未找到数据'))
        law_content = format_data(json_item.get('law_articles', '未找到数据'))  # <--- 新增的具体法条内容
        pred_laws = format_data(json_item.get('law_numbers', '未找到数据'))
        
        resp_aa = format_data(json_item.get('response_aa', '未找到数据'))
        resp_no_rag = format_data(json_item.get('response_analysis_no_rag', '未找到数据'))
        sim_cases = format_data(json_item.get('similar_cases', '未找到数据'))
        
        row_data = {'Case_Index': idx}
        for dim in dims:
            row_data[f'LLM_{dim}'] = model_df.loc[idx, dim]
        row_data['LLM_总分'] = model_df.loc[idx, '总分']
        llm_data.append(row_data)
        
        html_content += f"""
        <div class="case-container" data-idx="{idx}">
            <div class="case-header">题目 #{idx} (共 {SAMPLE_SIZE} 题的第 {i+1} 题)</div>
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
        <button class="submit-btn" onclick="downloadCSV()">✅ 全部打完，点击下载打分结果</button>

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
                alert("⚠️ 提示：您还有题目没有打完分（4个维度都要选哦），请检查是否有遗漏！");
                return;
            }
            
            let blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
            let link = document.createElement("a");
            let url = URL.createObjectURL(blob);
            link.setAttribute("href", url);
            link.setAttribute("download", "my_human_scores1.csv");
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            alert("🎉 下载成功！请将下载的 'my_human_scores1.csv' 与计算相关性的脚本放在一起跑一下。");
        }
        </script>
    </body>
    </html>
    """
    
    pd.DataFrame(llm_data).to_csv(llm_reference, index=False, encoding='utf-8-sig')
    with open(html_output, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"\n✅ 网页生成完毕！（已抽取 {SAMPLE_SIZE} 条，包含具体法条内容）")
    print(f"👉 请双击打开 【{html_output}】 开始愉快的阅卷吧！")

if __name__ == "__main__":
    main()