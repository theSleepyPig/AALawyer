import json
import random

input_path = "/mnt/ssd_2/yxma/LeLLM/data/data/data_valid.json"  
output_path = "aieval_dataset_200.json"  
num_samples = 200


with open(input_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

print(f"总数据条数：{len(data)}")

sample_size = min(num_samples, len(data))
sampled_data = random.sample(data, sample_size)

with open(output_path, "w", encoding="utf-8") as f_out:
    for item in sampled_data:
        f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"已保存 {sample_size} 条样本至：{output_path}")
