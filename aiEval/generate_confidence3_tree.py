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
# model_path = "/mnt/ssd_2/yxma/LeLLM/DeepSeek-R1-Distill-Qwen-7B/"

# ================= 模型加载 =================
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transformers.logging.set_verbosity_error()
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# ================= 硬约束类 (新增) =================

class LegalFormatConstraint:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
        # 预先获取关键字符的 Token ID
        # 使用 encode 取最后一个 token，确保拿到模型词表里对应的 ID
        def get_id(s):
            return tokenizer.encode(s, add_special_tokens=False)[-1]
        
        self.t_lb = get_id("[")
        self.t_rb = get_id("]")
        self.t_fa = get_id("法")
        self.t_tiao = get_id("条")
        self.t_xing = get_id("刑")
        self.t_di = get_id("第")
        self.t_sep = get_id("、")
        self.t_lt = get_id("<")
        self.t_e = get_id("e")
        self.t_o = get_id("o")
        self.t_a = get_id("a")
        self.t_gt = get_id(">")
        self.t_eos = tokenizer.eos_token_id
        
        # 数字 0-9
        self.digit_tokens = []
        for i in range(10):
            self.digit_tokens.append(get_id(str(i)))
            
    def get_constraint_fn(self, prompt_len):
        """返回给 generate 使用的约束函数"""
        
        def prefix_allowed_tokens_fn(batch_id, input_ids):
            # 获取当前新生成的内容
            # input_ids[prompt_len:] 是模型生成的 tokens
            gen_ids = input_ids[prompt_len:]
            gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            
            # === 状态机逻辑 ===
            
            # 1. 刚开始，必须 "["
            if len(gen_text) == 0:
                return [self.t_lb]
            
            # 2. 补全 "[法条]"
            if gen_text == "[": return [self.t_fa]
            if gen_text == "[法": return [self.t_tiao]
            if gen_text == "[法条": return [self.t_rb]
            
            # 3. 补全 "刑法第" (出现在 "]" 或 "、" 之后)
            # 逻辑：如果刚结束前缀或分隔符，必须接 "刑"
            if gen_text.endswith("]") or gen_text.endswith("、"):
                return [self.t_xing]
            
            # 处理 "刑" -> "法" -> "第"
            # 注意：用 endswith 检测，兼容前面已经有内容的情况
            if gen_text.endswith("刑"): return [self.t_fa]
            if gen_text.endswith("刑法"): return [self.t_di]
            
            # 4. 数字处理 (在 "第" 之后，或 数字 之后)
            if gen_text.endswith("刑法第"):
                return self.digit_tokens # 刚写完 "第"，必须写数字
            
            # 检查最后一段是否是数字
            # 提取最后一次出现的 "第" 之后的内容
            if "刑法第" in gen_text:
                last_segment = gen_text.split("刑法第")[-1]
                # 如果是纯数字 (e.g. "128")
                if len(last_segment) > 0 and last_segment.isdigit():
                    # 允许继续写数字，或者写 "条"
                    return self.digit_tokens + [self.t_tiao]
            
            # 5. "条" 之后，允许 "、" (继续) 或 "<" (准备结束)
            if gen_text.endswith("条"):
                return [self.t_sep, self.t_lt]
            
            # 6. <eoa> 补全逻辑
            # 一旦出现 "<"，强制按顺序生成 e -> o -> a -> >
            if gen_text.endswith("<"): return [self.t_e]
            if gen_text.endswith("<e"): return [self.t_o]
            if gen_text.endswith("<eo"): return [self.t_a]
            if gen_text.endswith("<eoa"): return [self.t_gt]
            
            # 7. 写完 <eoa>，强制 EOS
            if gen_text.endswith("<eoa>"):
                return [self.t_eos]
                
            # 兜底：如果状态跑飞了（理论上被上面逻辑锁死不会跑飞），允许结束
            return [self.t_eos]

        return prefix_allowed_tokens_fn

# ================= 其他辅助函数 =================

def parse_ground_truth(answer):
    """
    解析 Ground Truth
    """
    clean_text = answer.replace("法条:刑法第", "").replace("条", "").replace("法条:", "").replace("刑法", "")
    parts = clean_text.split("、")
    nums = []
    for p in parts:
        found = re.findall(r"\d+", p)
        if found:
            nums.extend(found)
    return nums

def extract_pred_numbers(text):
    """ 提取预测数字 """
    return re.findall(r"\d+", text)

# 初始化约束器
constraint_handler = LegalFormatConstraint(tokenizer)

def predict_with_confidence(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    # 获取 Prompt 的 token id，用于计算 prompt_len
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    prompt_len = model_inputs.input_ids.shape[1]

    # 获取当前 Prompt 的硬约束函数
    prefix_fn = constraint_handler.get_constraint_fn(prompt_len)

    outputs = model.generate(
        **model_inputs, 
        max_new_tokens=128, 
        do_sample=False, 
        return_dict_in_generate=True,
        output_scores=True,
        # 【核心修正】注入硬约束函数
        prefix_allowed_tokens_fn=prefix_fn
    )
    
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )
    
    # 简单的置信度计算 (平均 log prob)
    # 注意：如果生成序列为空(被约束卡死)，这里要处理异常
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
print(f"Total samples: {len(data)}. Starting inference with Hard Constraint...")

for idx, item in enumerate(tqdm(data)): 
    
    prompt = item["instruction"] + "\n" + item["question"]
    
    try:
        pred_text, confidence = predict_with_confidence(prompt)
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

# 保存
os.makedirs(os.path.dirname(output_json), exist_ok=True)
with open(output_json, "w", encoding="utf-8") as fout:
    json.dump(results, fout, ensure_ascii=False, indent=2)

if len(results) > 0:
    avg_f1 = sum(r['f1_score'] for r in results) / len(results)
else:
    avg_f1 = 0.0

print(f"\nResults saved to {output_json}")
print(f"Current F1 Score: {avg_f1:.4f}")