import json
import re
import os
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ================= 配置 =================
input_json = "/home/yxma/hzx/LeLLM/LawBench/data/zero_shot/3-1.json"  
output_json = "results/scr_confidence_3_1_fixed.json"
model_path = "/mnt/ssd_2/yxma/LeLLM/train_mergem20"

# ================= 模型加载 =================
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transformers.logging.set_verbosity_error()
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# ================= 核心修正函数 =================

def parse_ground_truth(answer):
    """
    复刻 ljp_article.py 的解析逻辑
    输入: "法条:刑法第264条" 或 "法条:刑法第234、275条"
    输出: ['264'] 或 ['234', '275']
    """
    # 1. 清洗前缀后缀
    clean_text = answer.replace("法条:刑法第", "").replace("条", "").replace("法条:", "").replace("刑法", "")
    # 2. 按顿号分割
    parts = clean_text.split("、")
    # 3. 提取数字
    nums = []
    for p in parts:
        found = re.findall(r"\d+", p)
        if found:
            nums.extend(found)
    return nums

def extract_pred_numbers(text):
    """ 提取预测结果中的数字 """
    # 简单粗暴提取所有数字，防止模型输出格式怪异
    # 如果想更严谨，可以先定位 [法条]...<eoa>
    return re.findall(r"\d+", text)

def predict_with_confidence(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    outputs = model.generate(
        **model_inputs, 
        max_new_tokens=128, 
        do_sample=False, 
        return_dict_in_generate=True,
        output_scores=True
    )
    
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )
    avg_log_prob = torch.mean(transition_scores[0])
    confidence = torch.exp(avg_log_prob).item()

    input_len = model_inputs.input_ids.shape[1]
    generated_text = tokenizer.decode(outputs.sequences[:, input_len:][0], skip_special_tokens=True)
    
    return generated_text.strip(), confidence

# ================= 主流程 =================

with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)

results = []
print(f"Total samples: {len(data)}. Starting inference...")

# 建议跑全量，或者至少跑前 500 个
for idx, item in enumerate(tqdm(data)): 
    
    # 【关键修正1】Prompt 构造：直接使用 instruction + question (包含罪名)
    # 保持与 evab.py 逻辑完全一致
    prompt = item["instruction"] + "\n" + item["question"]
    
    # 预测
    pred_text, confidence = predict_with_confidence(prompt)
    
    # 【关键修正2】解析逻辑升级
    raw_answer = item.get("answer", "")
    ground_truth = parse_ground_truth(raw_answer)
    pred_articles = extract_pred_numbers(pred_text)
    
    # 计算 F1
    pred_set = set(pred_articles)
    true_set = set(ground_truth)
    
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    results.append({
        "id": idx,
        "input": prompt,
        "raw_output": pred_text,
        "pred_articles": pred_articles,
        "true_articles": ground_truth,
        "confidence": confidence,
        "f1_score": f1
    })

# 保存
os.makedirs(os.path.dirname(output_json), exist_ok=True)
with open(output_json, "w", encoding="utf-8") as fout:
    json.dump(results, fout, ensure_ascii=False, indent=2)

avg_f1 = sum(r['f1_score'] for r in results) / len(results)
print(f"\nResults saved to {output_json}")
print(f"Current F1 Score: {avg_f1:.4f}")