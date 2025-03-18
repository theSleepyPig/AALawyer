import json
import re

import cn2an

# 输入文件路径
input_file = "/mnt/ssd_2/yxma/LeLLM/data/data/data_task1_only.json"
output_file = "/mnt/ssd_2/yxma/LeLLM/data/data/RAGDatabase1.json"

# 读取 JSON 文件
with open(input_file, "r", encoding="utf-8") as file:
    data = json.load(file)

# 处理数据
law_dict = {}

for entry in data:
    text = entry["text"].strip()
    split_text = text.split("  ", 1)  # 以双空格分割编号和内容
    if len(split_text) == 2:
        law_number = split_text[0].strip()
        law_content = split_text[1].strip()
        
        # # 只保留编号中的数字
        # number_match = re.search(r"\d+", law_number)
        # if number_match:
        #     law_number = number_match.group()

        # law_dict[law_number] = law_content
                # 提取中文数字部分
        number_match = re.search(r"第([一二三四五六七八九十百千万零〇]+)条", law_number)
        if number_match:
            chinese_number = number_match.group(1)
            # 转换为阿拉伯数字
            arabic_number = cn2an.transform(chinese_number, "cn2an")
            law_dict[str(arabic_number)] = law_content

# 保存处理后的数据
with open(output_file, "w", encoding="utf-8") as outfile:
    json.dump(law_dict, outfile, ensure_ascii=False, indent=4)

print(f"处理完成，结果已保存到 {output_file}")
