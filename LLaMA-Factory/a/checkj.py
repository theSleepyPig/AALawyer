import json

file_path = "/home/yxma/hzx/hzx/LeLLM/ckpt/data/data_task1_unsupervised.jsonl"

with open(file_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            print(f"JSON 解析错误: 第 {i+1} 行 -> {e}")