import json
import os
import torch
import argparse
from tqdm import tqdm
from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download
import transformers
import wandb

# 解析命令行参数
parser = argparse.ArgumentParser(description="Evaluate LLM on Multiple Legal Case Analysis Tasks")
parser.add_argument("--experiment_id", type=str, default="m03", help="Experiment ID, used to set model path")
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

# 设置模型下载路径
model_id = "wisdomOcean/wisdomInterrogatory"
revision = "v1.0.0"
# 指定下载路径
model_dir = "/mnt/ssd_2/yxma/LeLLM/wisdomInterrogatory"  # 修改为你希望存放模型的路径

# 如果模型目录不存在，则下载模型
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    snapshot_download(model_id, revision, cache_dir=model_dir)

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype="auto",
    device_map=device_map_setting  # 动态设备映射
)
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

# 任务列表
task_list = [
    "3-8", "3-2", "3-4", "3-5", "3-7", "2-5", "2-6",
    "2-7", "2-1", "2-2", "2-3", "2-4", "2-9", "2-10", "1-1", "1-2"
]

# 根据 mode 选择 JSON 数据目录
json_base_path = f"/home/yxma/hzx/hzx/LeLLM/LawBench/data/{args.mode}/"
output_base_folder = f"/home/yxma/hzx/hzx/LeLLM/ckpt/predictions/{args.mode}/output_results_{args.experiment_id}"

# 生成答案
def generate_response(prompt):
    inputs = tokenizer(f'<s>Human: {prompt} </s>Assistant: ', return_tensors='pt')
    inputs = inputs.to(model.device)  # 将输入移动到正确的设备

    # 生成预测结果
    pred = model.generate(**inputs, max_new_tokens=args.max_new_tokens, repetition_penalty=1.2)

    # 解码并返回生成的回答
    response = tokenizer.decode(pred[0], skip_special_tokens=True)
    return response.split("Assistant: ")[1]  # 获取回答部分

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
