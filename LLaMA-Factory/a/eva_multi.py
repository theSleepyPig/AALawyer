# python a/eva_multi.py
# nohup python a/eva.py > eva28m12.log 2>&1 &

import json
import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import transformers

import wandb


# para
experiment_id = "m12"
device_map_setting = {"": 3} 
# device_map_setting = {0: 1, 1: 2}
test_nums = ["2-8", "2-9", "2-10"]  # 需要处理的测试任务编号
# test_num="2-8"


# 初始化 WandB 监控
wandb.init(project=f"-Test-{experiment_id}-batch", entity='xuan_LeLLM', reinit=True)
print(f"Process ID: {os.getpid()}")

transformers.logging.set_verbosity_error()  # 只显示错误，不显示警告

# 选择模型路径
if experiment_id == "m0":
    model_path = "/mnt/ssd_2/yxma/LeLLM/DeepSeek-R1-Distill-Qwen-7B/"
else:
    model_path = f"/mnt/ssd_2/yxma/LeLLM/train_merge{experiment_id}/"

print(f"Experiment: {experiment_id}")
print(f"Model Path: {model_path}")

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map=device_map_setting  # 指定 GPU
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 结果保存路径
output_folder = f"/home/yxma/hzx/hzx/LeLLM/ckpt/predictions/zero_shot/output_results_{experiment_id}"
os.makedirs(output_folder, exist_ok=True)

def generate_response(prompt):
    """
    生成模型回答
    """
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 生成响应
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response.strip()

def process_test_file(json_file_path, test_num):
    """
    处理单个测试 JSON 文件，并单独保存结果
    """
    with open(json_file_path, "r", encoding="utf-8") as file:
        input_data = json.load(file)
    
    results = {}

    for idx, item in enumerate(tqdm(input_data, desc=f"Processing {test_num}", unit="case")):
        # 构造 Prompt
        prompt = item["instruction"] + "\n" + item["question"]

        # 生成模型预测结果
        prediction = generate_response(prompt)

        # 记录结果
        results[str(idx)] = {
            "origin_prompt": prompt,
            "prediction": prediction,
            "refr": item.get("answer", "")  # 参考答案
        }

    # 保存到单独的文件
    output_file_path = os.path.join(output_folder, f"{test_num}.json")
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=4, ensure_ascii=False)

    print(f"✅ 任务 {test_num} 完成，结果已保存至 {output_file_path}")

def run_all_tests(test_nums):
    """
    依次处理多个测试任务，并分别保存结果
    """
    for test_num in test_nums:
        json_file_path = f"/home/yxma/hzx/hzx/LeLLM/LawBench/data/zero_shot/{test_num}.json"
        
        if not os.path.exists(json_file_path):
            print(f"⚠️ 文件 {json_file_path} 不存在，跳过...")
            continue
        
        process_test_file(json_file_path, test_num)

# 运行所有测试任务
run_all_tests(test_nums)