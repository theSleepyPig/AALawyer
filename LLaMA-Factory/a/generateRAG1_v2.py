import json
import re
import cn2an

# 输入输出路径
input_file = "/mnt/ssd_2/yxma/LeLLM/data/data/data_task1_only.json"
output_file = "/mnt/ssd_2/yxma/LeLLM/data/data/RAGDatabase1_v2.json"

# 读取 JSON 数据
with open(input_file, "r", encoding="utf-8") as file:
    data = json.load(file)

# 初始化法条字典
law_dict = {}

# 遍历数据
for entry in data:
    text = entry["text"].strip()
    split_text = text.split("  ", 1)  # 按双空格分离编号和内容
    if len(split_text) == 2:
        law_number = split_text[0].strip()
        law_content = split_text[1].strip()

        # 匹配“第十四条”、“第十四条之一”、“第十四条之二”……
        match = re.search(r"第([一二三四五六七八九十百千万零〇]+)条(?:之[一二三四五六七八九十百千万零〇]+)?", law_number)
        if match:
            chinese_number = match.group(1)
            arabic_number = cn2an.transform(chinese_number, "cn2an")
            key = str(arabic_number)

            # 将所有子条拼接到主条中
            if key not in law_dict:
                law_dict[key] = ""

            law_dict[key] += f"{law_number} {law_content}\n"

# 去除每条最后一个换行符
for key in law_dict:
    law_dict[key] = law_dict[key].rstrip()

# 保存结果
with open(output_file, "w", encoding="utf-8") as outfile:
    json.dump(law_dict, outfile, ensure_ascii=False, indent=4)

print(f"处理完成，结果已保存到 {output_file}")
