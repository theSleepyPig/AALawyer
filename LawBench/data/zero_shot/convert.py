import json

# 读取你的 JSON 文件
with open("1-2.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 转换为 `sharegpt` 格式
converted_data = []
for item in data:
    new_entry = {
        "messages": [
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": f"[正确答案]{item['answer']}<eoa>"}
        ]
    }
    converted_data.append(new_entry)

# 保存
with open("1-2_converted.json", "w", encoding="utf-8") as f:
    json.dump(converted_data, f, indent=4, ensure_ascii=False)

print("转换完成")