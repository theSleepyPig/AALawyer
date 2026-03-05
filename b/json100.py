import json

# 原始 JSON 文件路径
input_file = '/mnt/ssd_2/yxma/LeLLM/data/data/merge.json'

# 输出（裁剪后）JSON 文件路径
output_file = '/mnt/ssd_2/yxma/LeLLM/data/data/CCs_sample.json'

# 读取 JSON 数据
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取前 100 个元素
first_100 = data[:100]

# 保存到新文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(first_100, f, ensure_ascii=False, indent=2)

print(f"前100条数据已保存到: {output_file}")
