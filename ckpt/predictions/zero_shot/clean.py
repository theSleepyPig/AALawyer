import os
import json
import re

# 输入和输出的文件夹路径
input_folder = '/home/yxma/hzx/hzx/LeLLM/ckpt/predictions/zero_shot/output_results_m13_think'  # 输入文件夹路径
output_folder = '/home/yxma/hzx/hzx/LeLLM/ckpt/predictions/zero_shot/output_results_m13'  # 输出文件夹路径

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 获取目录下的所有文件
files = os.listdir(input_folder)

# 处理每个文件
for file_name in files:
    file_path = os.path.join(input_folder, file_name)

    # 只处理 json 文件
    if file_name.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 删除 prediction 字段中的 <think> 部分
        for entry in data.values():
            # 使用正则表达式删除 <think> 标签及其内容
            entry['prediction'] = re.sub(r'<think>.*?</think>', '', entry['prediction'], flags=re.DOTALL).strip()

        # 将处理后的数据保存到新文件夹
        output_file_path = os.path.join(output_folder, file_name)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

print("处理完成！")
