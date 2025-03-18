# import pandas as pd
# import json

# # 读取 CSV 文件
# df = pd.read_csv("/home/yxma/hzx/hzx/LeLLM/ckpt/data/data_task1.csv")

# # 确保 CSV 文件只有一列，并转换成 JSONL 格式
# df.to_json("/home/yxma/hzx/hzx/LeLLM/ckpt/data/data_task1.jsonl", orient="records", lines=True, force_ascii=False)

# import json

# input_file = "/home/yxma/hzx/hzx/LeLLM/ckpt/data/data_task1.jsonl"
# output_file = "/home/yxma/hzx/hzx/LeLLM/ckpt/data/data_task1_unsupervised.jsonl"

# with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
#     for line in f_in:
#         data = json.loads(li
#         new_data = {
#             "text": data["all"]
#         }
#         f_out.write(json.dumps(new_data, ensure_ascii=False) + "\n")

# print("转换完成，数据已保存至:", output_file)

import json

input_file = "/home/yxma/hzx/hzx/LeLLM/ckpt/data/data_task1_unsupervised.jsonl"
output_file = "/home/yxma/hzx/hzx/LeLLM/ckpt/data/data_task1_fixed.json"

# 读取 JSONL 文件并格式化为 JSON 数组
with open(input_file, "r", encoding="utf-8") as fin:
    data = [json.loads(line.strip()) for line in fin if line.strip()]

# 写入新的 JSON 文件，前后加中括号，每行后面加逗号（最后一个对象除外）
with open(output_file, "w", encoding="utf-8") as fout:
    fout.write("[\n")
    for i, obj in enumerate(data):
        json.dump(obj, fout, ensure_ascii=False, indent=4)
        if i != len(data) - 1:
            fout.write(",\n")  # 逗号用于分隔 JSON 对象，最后一个对象不加
    fout.write("\n]")

print(f"格式化完成，已保存至 {output_file}")