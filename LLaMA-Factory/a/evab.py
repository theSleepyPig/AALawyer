# nohup python /home/yxma/hzx/hzx/LeLLM/LLaMA-Factory/a/evab.py \
#     --experiment_id m22 \
#     --device 1 \
#     --mode zero_shot \
#     --max_new_tokens 2048 \
#     > evabash_zero_shot_m22.log 2>&1 &

# nohup python /home/yxma/hzx/hzx/LeLLM/LLaMA-Factory/a/evab.py \
#     --experiment_id m13 \
#     --device 1 \
#     --mode zero_shot \
#     --max_new_tokens 4096 \
#     > evabash_zero_shot_m13.log 2>&1 &

import json
import os
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import wandb

# 解析命令行参数
parser = argparse.ArgumentParser(description="Evaluate LLM on Multiple Legal Case Analysis Tasks")
parser.add_argument("--experiment_id", type=str, default="m20", help="Experiment ID, used to set model path")
parser.add_argument("--device", type=str, default="0", help="CUDA device(s) to use, e.g., '0' or '0,1,2'")
parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum new tokens to generate")
parser.add_argument("--mode", type=str, choices=["zero_shot", "one_shot"], default="zero_shot", help="Evaluation mode: zero_shot or one_shot")
args = parser.parse_args()

# 设备映射（解析多个 GPU）
device_list = [int(d) for d in args.device.split(",")]
device_map_setting = {str(i): device_list[i] for i in range(len(device_list))} if len(device_list) > 1 else {"": device_list[0]}

# W&B 初始化
wandb.init(
    project=f"LegalLLM-Test-{args.experiment_id}",
    entity='xuan_LeLLM',
    config={"mode": args.mode},  # 记录超参
    reinit=True
)
print(f"Process ID: {os.getpid()}")

# 只显示错误，不显示警告
transformers.logging.set_verbosity_error()

# 选择模型路径
if args.experiment_id == "m0":
    model_path = "/mnt/ssd_2/yxma/LeLLM/DeepSeek-R1-Distill-Qwen-7B/"
elif args.experiment_id == "m02":
    model_path = "/mnt/ssd_2/yxma/LeLLM/Qwen2.5-Math-7B/"
else:
    model_path = f"/mnt/ssd_2/yxma/LeLLM/train_merge{args.experiment_id}/"
print(f"Model Path: {model_path}")

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map=device_map_setting  # 动态设备映射
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 任务列表
task_list = [
    "3-1", "3-3", "3-8", 
    "3-7", "2-5", "2-6",   
    "2-7","2-2", "2-3", "2-4", 
    "2-9"
]
# task_list = [
#     "2-8", "3-1", "3-3", "3-6", "3-8", 
#     "3-2", "3-4", "3-5", "3-7", "2-5", "2-6", "2-7", "2-1", "2-2", "2-3", "2-4", "2-9", "2-10", "1-1", "1-2"
# ]

# 根据 mode 选择 JSON 数据目录
json_base_path = f"/home/yxma/hzx/hzx/LeLLM/LawBench/data/{args.mode}/"
output_base_folder = f"/home/yxma/hzx/hzx/LeLLM/ckpt/predictions/{args.mode}/output_results_{args.experiment_id}"

# 生成答案
def generate_response(prompt):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=args.max_new_tokens)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response.strip()

# 读取 JSON 数据并测试
def test_model_from_json(json_file_path, output_folder, test_num):
    try:
        with open(json_file_path, "r", encoding="utf-8") as file:
            input_data = json.load(file)
    except FileNotFoundError:
        print(f"Skipping {test_num}: JSON file not found.")
        return
    
    results = {}

    for idx, item in enumerate(tqdm(input_data, desc=f"Processing {test_num} ({args.mode})", unit="case")):
        prompt = item["instruction"] + "\n" + item["question"]
        prediction = generate_response(prompt)

        results[str(idx)] = {
            "origin_prompt": prompt,
            "prediction": prediction,
            "refr": item.get("answer", "")
        }

    os.makedirs(output_folder, exist_ok=True)
    output_file_path = os.path.join(output_folder, f"{test_num}.json")

    with open(output_file_path, "w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=4, ensure_ascii=False)

# 遍历所有任务
for task_num in task_list:
    json_file_path = os.path.join(json_base_path, f"{task_num}.json")
    test_model_from_json(json_file_path, output_base_folder, task_num)

