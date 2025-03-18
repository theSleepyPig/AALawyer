import json

input_file = "/home/yxma/hzx/hzx/LeLLM/ckpt/data/data_task2.csv"  # 你的文本文件路径
output_file = "/home/yxma/hzx/hzx/LeLLM/ckpt/data/data_task2_converted.json"

# 读取文本文件，并转换为 JSON 结构
with open(input_file, "r", encoding="utf-8") as fin:
    lines = fin.read().strip().split("\n")  # 按换行符分割

# 过滤掉空行，并转换为 JSON 数组
data = [{"text": line.strip()} for line in lines if line.strip()]

# 格式化输出 JSON，每个对象后面带逗号，最后一个对象不加
with open(output_file, "w", encoding="utf-8") as fout:
    fout.write("[\n")
    for i, obj in enumerate(data):
        json.dump(obj, fout, ensure_ascii=False, indent=4)
        if i != len(data) - 1:
            fout.write(",\n")  # 逗号分隔 JSON 对象，最后一个不加
    fout.write("\n]")

print(f"转换完成，已保存至 {output_file}")
