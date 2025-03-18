import json
import os
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import wandb

# 解析命令行参数
parser = argparse.ArgumentParser(description="Evaluate LLM on Legal Case Analysis Task")
parser.add_argument("--experiment_id", type=str, default="m20", help="Experiment ID, used to set model path")
parser.add_argument("--test_num", type=str, default="3-6", help="Test set identifier")
parser.add_argument("--device", type=str, default="0", help="CUDA device(s) to use, e.g., '0' or '0,1,2'")
parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum new tokens to generate")
args = parser.parse_args()

# 设备映射（解析多个 GPU）
device_list = [int(d) for d in args.device.split(",")]
device_map_setting = {str(i): device_list[i] for i in range(len(device_list))} if len(device_list) > 1 else {"": device_list[0]}

# W&B 初始化
wandb.init(project=f"-Test-{args.experiment_id}-{args.test_num}", entity='xuan_LeLLM', reinit=True)
print(f"Process ID: {os.getpid()}")

# 只显示错误，不显示警告
transformers.logging.set_verbosity_error()

# 选择模型路径
if args.experiment_id == "m0":
    model_path = "/mnt/ssd_2/yxma/LeLLM/DeepSeek-R1-Distill-Qwen-7B/"
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

# JSON 文件路径
json_file_path = f"/home/yxma/hzx/hzx/LeLLM/LawBench/data/zero_shot/{args.test_num}.json"
output_folder = f"/home/yxma/hzx/hzx/LeLLM/ckpt/predictions/zero_shot/output_results_{args.experiment_id}"
print(f"Output Folder: {output_folder}")

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
    with open(json_file_path, "r", encoding="utf-8") as file:
        input_data = json.load(file)
    
    results = {}

    for idx, item in enumerate(tqdm(input_data, desc="Processing cases", unit="case")):
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

# 运行测试
test_model_from_json(json_file_path, output_folder, args.test_num)
