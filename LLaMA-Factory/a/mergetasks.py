import json
import random

# 输入文件路径
input_files = [
    # "/mnt/ssd_2/yxma/LeLLM/data/data/data_train_sft/articles.json",
    "/mnt/ssd_2/yxma/LeLLM/data/data/data_train_sft/accusations.json",
    "/mnt/ssd_2/yxma/LeLLM/data/data/data_train_sft/fines.json",
    "/mnt/ssd_2/yxma/LeLLM/data/data/data_train_sft/criminals.json",
    "/mnt/ssd_2/yxma/LeLLM/data/data/data_train_sft/sentences.json",
    # "/home/yxma/hzx/hzx/LeLLM/ckpt/data/DISC/DISC-Law-SFT-Pair_train.json",
    # "/home/yxma/hzx/hzx/LeLLM/ckpt/data/DISC/DISC-Law-SFT-Pair-QA-released_train.json",
    # "/home/yxma/hzx/hzx/LeLLM/ckpt/data/DISC/DISC-Law-SFT-Triplet-QA-released_train.json",
    # "/home/yxma/hzx/hzx/LeLLM/ckpt/data/DISC/DISC-Law-SFT-Triplet-released_train.json",
    "/mnt/ssd_2/yxma/LeLLM/data/data/DISC/DISC-Law-SFT-Pair-QA-released_train.json",
    "/mnt/ssd_2/yxma/LeLLM/data/data/DISC/DISC-Law-SFT-Pair_train.json",
    "/mnt/ssd_2/yxma/LeLLM/data/data/DISC/DISC-Law-SFT-Triplet-QA-released_train.json",
    "/mnt/ssd_2/yxma/LeLLM/data/data/DISC/DISC-Law-SFT-Triplet-released_train.json"
]

# 输出文件路径
output_file = "/mnt/ssd_2/yxma/LeLLM/data/data/data_train_sft/sftdata_exclude_articles.json"

# 存储所有数据
merged_data = []

# 逐个读取文件并合并数据
for file in input_files:
    with open(file, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)  # 读取 JSON
            merged_data.extend(data)  # 合并数据
        except json.JSONDecodeError as e:
            print(f"解析失败: {file}, 错误: {e}")

# 随机打乱数据
random.shuffle(merged_data)

# 写入合并后的 JSON 文件
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=4)

print(f"数据合并完成，共 {len(merged_data)} 条数据，已保存至 {output_file}！")
