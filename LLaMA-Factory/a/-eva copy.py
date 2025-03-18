# python a/eva.py
# nohup python a/eva.py > evam20.log 2>&1 &

import json
import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import transformers

import wandb


# para
experiment_id = "m20"
device_map_setting = {"": 3} 
# device_map_setting = {0: 1, 1: 2}
test_num="3-6"


wandb.init(project=f"-Test-{experiment_id}-{test_num}", entity='xuan_LeLLM', reinit=True)
print(f"Process ID: {os.getpid()}")


transformers.logging.set_verbosity_error()  # 只显示错误，不显示警告

# base模型
if experiment_id == "m0":
    model_path = "/mnt/ssd_2/yxma/LeLLM/DeepSeek-R1-Distill-Qwen-7B/"
else:
    # 预训练过法条数据-1epoch
    model_path = f"/mnt/ssd_2/yxma/LeLLM/train_merge{experiment_id}/"
print(experiment_id)
print(model_path)

# 加载本地模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map=device_map_setting  # 这里指定单卡 cuda:0
    # device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 示例：指定 JSON 文件路径和输出文件夹
json_file_path = f"/home/yxma/hzx/hzx/LeLLM/LawBench/data/zero_shot/{test_num}.json"
output_folder = f"/home/yxma/hzx/hzx/LeLLM/ckpt/predictions/zero_shot/output_results_{experiment_id}"  # +all article pretrain
print(output_folder)


def generate_response(prompt):
    """
    使用本地 model 生成答案
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

def test_model(input_data):
    """
    测试模型的输出，确保格式正确
    :param input_data: 输入数据，格式为列表，包含多个案例
    :return: 输出JSON格式的预测结果
    """
    results = {}

    for idx, item in enumerate(input_data):
        # 生成Prompt
        prompt = item["instruction"] + "\n" + item["question"]

        # 生成模型预测结果
        prediction = generate_response(prompt)

        # 记录结果
        results[str(idx)] = {
            "origin_prompt": prompt,
            "prediction": prediction,
            "refr": item.get("answer", "")  # 参考答案
        }

    return results

# # 加载本地模型和分词器
# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     torch_dtype="auto",
#     device_map={"": 0}  # 这里指定单卡 cuda:0
# )
# tokenizer = AutoTokenizer.from_pretrained(model_path)


def test_model_from_json(json_file_path, output_folder, test_num):
    """
    从 JSON 文件读取测试数据，并测试模型输出格式
    :param json_file_path: JSON 文件路径
    :param output_folder: 结果存放的文件夹
    """
    # 读取 JSON 文件
    with open(json_file_path, "r", encoding="utf-8") as file:
        input_data = json.load(file)
    
    results = {}

    for idx, item in enumerate(tqdm(input_data, desc="Processing cases", unit="case")):
        # 生成Prompt
        prompt = item["instruction"] + "\n" + item["question"]

        # 生成模型预测结果
        prediction = generate_response(prompt)

        # 记录结果
        results[str(idx)] = {
            "origin_prompt": prompt,
            "prediction": prediction,
            "refr": item.get("answer", "")  # 参考答案
        }

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 生成输出文件路径
    output_file_path = os.path.join(output_folder, f"{test_num}.json")
    
    # 将结果写入 JSON 文件
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=4, ensure_ascii=False)




# 运行测试
test_model_from_json(json_file_path, output_folder, test_num)

# # 示例输入
# input_data = [
#     {
#         "instruction": "根据下列事实和罪名给出涉及的刑法法条...",
#         "question": "事实: 公诉机关指控：2016年3月28日...",
#         "answer": "法条:刑法第264条"
#     },
#     {
#         "instruction": "根据下列事实和罪名给出涉及的刑法法条...",
#         "question": "事实: 永顺县人民检察院指控...",
#         "answer": "法条:刑法第236条"
#     }
# ]

# # 运行测试
# output_results = test_model(input_data)

# # 打印JSON格式的结果
# print(json.dumps(output_results, indent=4, ensure_ascii=False))

