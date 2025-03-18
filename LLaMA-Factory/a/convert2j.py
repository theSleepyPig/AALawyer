import json

def convert_jsonl_to_json(jsonl_file, json_file):
    # 读取 merge.jsonl 文件
    with open(jsonl_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
        
    # 将每一行解析为字典，并存储在一个列表中
    data = []
    for line in lines:
        data.append(json.loads(line))
    
    # 将数据写入到 merge.json 文件
    with open(json_file, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)

# 输入和输出文件路径
jsonl_file_path = '/mnt/ssd_2/yxma/LeLLM/data/data/merge.jsonl'  # 替换为实际的merge.jsonl文件路径
json_file_path = '/mnt/ssd_2/yxma/LeLLM/data/data/merge.json'    # 输出文件路径

# 调用函数进行转换
convert_jsonl_to_json(jsonl_file_path, json_file_path)

print("转换完成！")
