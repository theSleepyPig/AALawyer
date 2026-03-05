import json
import re
import os
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ================= 配置 (保持与 Hard Constraint 版本一致) =================
input_json = "/home/yxma/hzx/LeLLM/LawBench/data/zero_shot/3-1.json"  
output_json = "results/scr_confidence_3_1_baseline.json" # 修改输出文件名以区分
# model_path = "/mnt/ssd_2/yxma/LeLLM/train_mergem20"
model_path = "/mnt/ssd_2/yxma/LeLLM/DeepSeek-R1-Distill-Qwen-7B/"

# ================= 模型加载 =================
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transformers.logging.set_verbosity_error()
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# ================= 辅助函数 (保持完全一致) =================

def parse_ground_truth(answer):
    clean_text = answer.replace("法条:刑法第", "").replace("条", "").replace("法条:", "").replace("刑法", "")
    parts = clean_text.split("、")
    nums = []
    for p in parts:
        found = re.findall(r"\d+", p)
        if found:
            nums.extend(found)
    return nums

def extract_pred_numbers(text):
    return re.findall(r"\d+", text)

# ================= 预测函数 (无约束版) =================

def predict_without_constraint(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # 【差异点】：这里去掉了 prefix_allowed_tokens_fn
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
    
    if len(transition_scores[0]) > 0:
        avg_log_prob = torch.mean(transition_scores[0])
        confidence = torch.exp(avg_log_prob).item()
    else:
        confidence = 0.0

    input_len = model_inputs.input_ids.shape[1]
    generated_text = tokenizer.decode(outputs.sequences[:, input_len:][0], skip_special_tokens=True)
    
    return generated_text.strip(), confidence

# ================= 主流程 =================

with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)

results = []
print(f"Total samples: {len(data)}. Starting Baseline Inference (No Constraint)...")

for idx, item in enumerate(tqdm(data)): 
    
    prompt = item["instruction"] + "\n" + item["question"]
    
    try:
        # 调用无约束的预测函数
        pred_text, confidence = predict_without_constraint(prompt)
    except Exception as e:
        print(f"Error at idx {idx}: {e}")
        pred_text = ""
        confidence = 0.0
    
    raw_answer = item.get("answer", "")
    ground_truth = parse_ground_truth(raw_answer)
    pred_articles = extract_pred_numbers(pred_text)
    
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

# 保存结果
os.makedirs(os.path.dirname(output_json), exist_ok=True)
with open(output_json, "w", encoding="utf-8") as fout:
    json.dump(results, fout, ensure_ascii=False, indent=2)

if len(results) > 0:
    avg_f1 = sum(r['f1_score'] for r in results) / len(results)
else:
    avg_f1 = 0.0

print(f"\nBaseline Results saved to {output_json}")
print(f"Baseline F1 Score: {avg_f1:.4f}")