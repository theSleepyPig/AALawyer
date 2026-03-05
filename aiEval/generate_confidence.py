import json
import re
import os
import torch
import transformers
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm 

# ================= 路径设置 =================
# input_json = "/home/yxma/hzx/hzx/LeLLM/aiEval/aieval_dataset_10.json" 
input_json = "/home/yxma/hzx/LeLLM/aiEval/aieval_dataset_1000_t1.json" 
output_json = "results/scr_confidence_calibration_test.json"  # 专门存这个实验结果

model_path = "/mnt/ssd_2/yxma/LeLLM/train_mergem20"
# model_path = "/mnt/ssd_2/yxma/LeLLM/DeepSeek-R1-Distill-Qwen-7B"

# ================= 模型加载 =================
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transformers.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print(f"Loading model from {model_path} ...")
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# ================= 工具函数 =================

def extract_law_numbers(text):
    """ 
    从 LLM 生成的文本中提取法条编号
    例如: "[法条]刑法第232条、第133条<eoa>" -> ['232', '133']
    """
    # 只要是数字就提取，或者配合 "第"字提取，防止提取到其他的数字
    return re.findall(r"第(\d+)条", text)

def calculate_f1(pred_list, true_list):
    """
    计算 F1 Score
    """
    pred_set = set([str(x) for x in pred_list])
    true_set = set([str(x) for x in true_list])
    
    if len(true_set) == 0:
        return 0.0 # 避免除零，理论上数据集中都有标签
        
    tp = len(pred_set & true_set) # 预测对了几个
    fp = len(pred_set - true_set) # 多预测了几个
    fn = len(true_set - pred_set) # 漏预测了几个
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1

def predict_law_with_confidence(prompt):
    """ 
    生成法条预测，并计算置信度 (Sequence Confidence)
    """
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # 1. 生成 (Greedy Decoding)
    # 只需要生成很短的内容，max_new_tokens 不用很大
    outputs = model.generate(
        **model_inputs, 
        max_new_tokens=256, 
        do_sample=False, 
        return_dict_in_generate=True,
        output_scores=True
    )
    
    # 2. 获取生成的 token
    input_len = model_inputs.input_ids.shape[1]
    generated_tokens = outputs.sequences[:, input_len:]
    
    # 3. 计算 Confidence (核心)
    # transition_scores 是每个 token 的 log_softmax 值
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )
    
    # 取 log 概率的平均值 (几何平均 Geometric Mean)
    # 这比取 min (最小值) 更稳定，比取 sum (总和) 更不受长度影响
    avg_log_prob = torch.mean(transition_scores[0])
    confidence = torch.exp(avg_log_prob).item()

    # 4. 解码文本
    response_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    
    return response_text.strip(), confidence

# ================= 主流程 =================

results = []

# 读取数据
with open(input_json, "r", encoding="utf-8") as f_in:
    data = [json.loads(line) for line in f_in if line.strip()]

print("Starting Confidence Calibration Experiment...")

for idx, item in enumerate(tqdm(data, desc="Testing SCR")):
    user_input = item["fact"]
    ground_truth = item["meta"]["relevant_articles"] # list of int, e.g. [232, 133]

    # 构造 Prompt
    prompt_law = (
            "根据下列事实和罪名给出涉及的刑法法条。"
            "只需给出刑法法条编号，请将答案填在[法条]与<eoa>之间。\n"
            "例如[法条]刑法第128条、刑法第341条<eoa>\n"
            f"事实: {user_input}\n"
        )
    
    # 运行预测
    prediction_text, confidence = predict_law_with_confidence(prompt_law)
    
    # 解析结果
    pred_law_numbers = extract_law_numbers(prediction_text)
    
    # 计算准确率指标
    f1_score = calculate_f1(pred_law_numbers, ground_truth)
    
    # 存下来
    results.append({
        "id": idx,
        "input": user_input,
        "raw_output": prediction_text,
        "pred_articles": pred_law_numbers,
        "true_articles": ground_truth,
        "confidence": confidence,  # <--- X轴
        "f1_score": f1_score       # <--- Y轴
    })

# 保存
os.makedirs(os.path.dirname(output_json), exist_ok=True)
with open(output_json, "w", encoding="utf-8") as fout:
    json.dump(results, fout, ensure_ascii=False, indent=2)

print(f"完成！结果已保存到 {output_json}")