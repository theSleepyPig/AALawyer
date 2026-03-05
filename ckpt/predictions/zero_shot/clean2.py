import os
import json
import re

# 要处理的输入文件路径列表
input_files = [
    '/home/yxma/hzx/LeLLM/ckpt/predictions/zero_shot/output_results_m0/4-1-t2.json',
    '/home/yxma/hzx/LeLLM/ckpt/predictions/zero_shot/output_results_m0/4-2-t2.json'
]

# 输出文件夹路径
output_folder = '/home/yxma/hzx/LeLLM/ckpt/predictions/zero_shot/output_results_m0'
os.makedirs(output_folder, exist_ok=True)

for file_path in input_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 删除 prediction 字段中的 <think> 标签及其内容
    for entry in data.values():
        entry['prediction'] = re.sub(r'<think>.*?</think>', '', entry['prediction'], flags=re.DOTALL).strip()

    # 替换文件名中的 -t.json 为 .json
    original_file_name = os.path.basename(file_path)
    cleaned_file_name = original_file_name.replace('-t2.json', '.json')
    output_file_path = os.path.join(output_folder, cleaned_file_name)

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

print("处理完成，文件名已重命名为 4-1.json 和 4-2.json")
