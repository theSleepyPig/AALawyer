#+similar cases
# nohup python /home/yxma/hzx/LeLLM/aiEval/generate_response_json_inter.py \
#     > generate_response_json_inter_run1_v2.log 2>&1 &
import csv
import json
import re
import os
import torch
import transformers
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from FlagEmbedding import FlagModel
from tqdm import tqdm 

# 路径设置
# input_json = "/home/yxma/hzx/hzx/LeLLM/aiEval/aieval_dataset_10.json" 
# output_json = "results/generated_10_v3_1.json"  # 改为 json 输出
input_json = "/home/yxma/hzx/LeLLM/aiEval/aieval_dataset_200_v2.jsonl" 
output_json = "results/generated_200_inter_v2023_run2.json"  # 改为 json 输出


results = []

# model_path = "/mnt/ssd_2/yxma/LeLLM/train_mergem20"
model_path = "/mnt/ssd_2/yxma/LeLLM/internlm3-8b-instruct/"
# model_path = "/mnt/ssd_2/yxma/LeLLM/DeepSeek-R1-Distill-Qwen-7B"
print(output_json)
print(model_path)


# 模型加载
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transformers.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print("正在加载模型和分词器 (InternLM3)...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    # torch_dtype=torch.float16,
    load_in_8bit=True,
    trust_remote_code=True,
    device_map="auto" # 直接映射到单个设备
)
# model.to(device)
print("模型已移动到 device:", model.device)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)
print("模型加载完成。")

# 法条数据库
# with open("/mnt/ssd_2/yxma/LeLLM/data/data/RAGDatabase1_v2.json", "r", encoding="utf-8") as file:
with open("/mnt/ssd_2/yxma/LeLLM/data/data/RAGDatabase_v2023_v3.json", "r", encoding="utf-8") as file:
    print("v3v3v3")
    law_data = json.load(file)

# FAISS 检索系统加载
bge_model = FlagModel(
    "/mnt/ssd_2/yxma/LeLLM/bge-large-zh",
    query_instruction_for_retrieval="为这个句子生成表示：",
    device="cuda:0",
    use_multi_process=False
)
faiss_index = faiss.read_index("/mnt/ssd_2/yxma/LeLLM/data/RAG2/legalCase_faiss.index")
with open("/mnt/ssd_2/yxma/LeLLM/data/RAG2/legalCase_ids.json", "r", encoding="utf-8") as f:
    index_ids = json.load(f)
with open("/mnt/ssd_2/yxma/LeLLM/data/data/merge.json", "r", encoding="utf-8") as f:
    index_texts = [r["text"].strip() for r in json.load(f)]

# 响应生成函数
def smart_translate(text):
    prompt = (
        "请将以下中文内容翻译为专业英文，保留法律术语和句式准确性。\n\n"
        f"中文原文：{text}\n\n"
        "英文翻译："
    )
    return generate_response(prompt)

def smart_translate2(text):
    segments = text.split("\n\n") if "\n\n" in text else [text]
    translated_parts = []
    for seg in segments:
        if not seg.strip():
            continue
        prompt = f"请将以下中文内容翻译为专业英文，保留法律术语和句式准确性。\n\n中文原文：{seg}\n\n英文翻译："
        result = generate_response(prompt)
        translated_parts.append(result.strip())
    return "\n\n".join(translated_parts)

def extract_law_numbers(text):
    """ 从 LLM 生成的文本中提取多个法条编号 """
    return re.findall(r"刑法第(\d+)条", text)

def retrieve_law_articles(law_numbers):
    """ 根据多个法条编号从 JSON 文件中检索对应的法条内容 """
    # articles = [f"刑法第{num}条: {law_data.get(num, '未找到相关法条')}" for num in law_numbers]
    articles = [f"{law_data.get(num, '未找到相关法条')}" for num in law_numbers]
    return "\n".join(articles) if articles else "未找到相关法条"

def translate(text, model, tokenizer):
    """ 翻译文本 """
    translated = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    translated_tokens = model.generate(**translated, max_length=512)
    # return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    translated_texts = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    return " ".join(translated_texts)

# def generate_response(prompt):
#     """ 使用 LLM 生成回复 """
#     messages = [
#         {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#         {"role": "user", "content": prompt}
#     ]
#     text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
#     print(f"🔢 输入 token 数量: {model_inputs['input_ids'].shape[1]}")
#     # print(f"✅ 模型最大输入 token 长度: {tokenizer.model_max_length}")
#     # print(tokenizer.truncation_side) 

#     generated_ids = model.generate(
#         **model_inputs, 
#         max_new_tokens=2048,
#         do_sample=False
#     )
    
#     generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

#     response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     return response.strip()

# def generate_response(prompt):
#     """ 使用 LLM 生成回复 """
#     # ✅ 改为手动拼接 Prompt
#     full_prompt = f"你是LawLLM，一个由复旦大学DISC实验室创造的法律助手。\n用户：{prompt}\n助手："
    
#     model_inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

#     # print(f"🔢 输入 token 数量: {model_inputs['input_ids'].shape[1]}")

#     generated_ids = model.generate(
#         **model_inputs,
#         max_new_tokens=2048,
#         do_sample=False
#     )

#     generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
#     response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
#     torch.cuda.empty_cache()
#     return response.strip()
def generate_response(prompt):
    # 构建 InternLM3 的标准 messages 格式
    torch.cuda.empty_cache()
    system_prompt = "You are an AI assistant for legal analysis."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    # 使用 apply_chat_template 生成 tokenized_chat
    tokenized_chat = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    input_ids = tokenized_chat
    batch_size, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len, dtype=torch.long, device=model.device).unsqueeze(0).repeat(batch_size, 1)

    # 模型生成
    generated_ids = model.generate(
        tokenized_chat,
        position_ids=position_ids,
        max_new_tokens=2048,
        do_sample=False
    )
    
    # 截取并解码新生成的内容
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(tokenized_chat, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    torch.cuda.empty_cache()
    return response.strip()
    
def retrieve_similar_cases(query, top_k=1):
    query_embedding = bge_model.encode([query])[0].astype("float32")
    D, I = faiss_index.search(np.array([query_embedding]), top_k)
    results = []
    for idx in I[0]:
        matched_text = index_texts[int(index_ids[idx])]
        results.append(matched_text)
    return results

# 执行主流程
with open(input_json, "r", encoding="utf-8") as f_in:
    data = [json.loads(line) for line in f_in if line.strip()]


for idx, item in enumerate(tqdm(data, desc="Generating answers:")):
    user_input = item["fact"]
    meta = item["meta"]

    #预测法条编号
    prompt_law = (
            "根据下列事实和罪名给出涉及的刑法法条。"
            "只需给出刑法法条编号，请将答案填在[法条]与<eoa>之间。\n"
            "例如[法条]刑法第128条、刑法第341条<eoa>\n"
            f"事实: {user_input}\n"
        )
    response_law_numbers = generate_response(prompt_law)
    # print(response_law_numbers)
    law_numbers = extract_law_numbers(response_law_numbers)
    # print(law_numbers)
    law_articles = retrieve_law_articles(law_numbers)

    # 不使用 RAG 的分析
    prompt_no_rag = (
            # "根据下列案件内容和相关法条分析案件，说明谁犯罪了，为什么认为他犯罪，涉及到哪条法律，犯了什么罪。分析在100字左右。分析时不需要提及法条内容，只提及相应编号。\n"
            
            f"案件内容: {user_input}\n"
            "根据以上案件内容，说明谁犯罪了，为什么认为他犯罪，涉及到哪条法律，犯了什么罪。结合案件内容和法条内容详细对比分析，把法条内容相和案件相关的整合进分析中。分析在100字左右。\n"
            
        )
    response_no_rag = generate_response(prompt_no_rag)
    
    # 只使用AA-RAG分析
    prompt_aa = (
            # "根据下列案件内容和相关法条分析案件，说明谁犯罪了，为什么认为他犯罪，涉及到哪条法律，犯了什么罪。分析在100字左右。分析时不需要提及法条内容，只提及相应编号。\n"
            
            f"案件内容: {user_input}\n"
            f"涉及法条内容:\n{law_articles}\n（涉及法条内容不一定都用上了，可能是这一条的某一部分）"
            "根据以上案件内容和相关法条分析案件，说明谁犯罪了，为什么认为他犯罪，涉及到哪条法律，犯了什么罪。结合案件内容和法条内容详细对比分析，把法条内容相和案件相关的整合进分析中。分析在100字左右。\n"
            
        )

    response_aa = generate_response(prompt_aa)
    

    #使用 RAG 的分析
    similar_cases = retrieve_similar_cases(user_input, top_k=1)
    prompt_rag = (
            f"案件内容: {user_input}\n"
            f"涉及法条内容:\n{law_articles}\n（涉及法条内容不一定都用上了，可能是这一条的某一部分）"
            "请根据案件内容、涉及的法条，以及如下提供的相似案例，综合分析案件。说明谁犯罪了，为什么认为他犯罪，涉及哪条法律，犯了什么罪。长度和格式参考相似案例。\n"
            # f"相似判决案例参考:\n{similar_cases}\n"
            f"【相似案例】:\n" + "\n\n".join(similar_cases) + "\n\n"
            
        )
    response_with_case = generate_response(prompt_rag)
    
    #仅使用 ccsRAG
    # similar_cases = retrieve_similar_cases(user_input, top_k=1)
    prompt_ccs = (
            f"案件内容: {user_input}\n"
            "请根据案件内容、涉及的法条，以及如下提供的相似案例，综合分析案件。说明谁犯罪了，为什么认为他犯罪，涉及哪条法律，犯了什么罪。长度和格式参考相似案例。\n"
            # f"相似判决案例参考:\n{similar_cases}\n"
            f"【相似案例】:\n" + "\n\n".join(similar_cases) + "\n\n"
            
        )
    response_only_ccs = generate_response(prompt_ccs)

    #构造结果字典
    result_item = {
        "input": user_input,
        "law_numbers": [f"刑法第{n}条" for n in law_numbers],
        "response_analysis_no_rag": response_no_rag,
        "response_aa": response_aa,
        "response_analysis_with_case": response_with_case,
        "response_analysis_onlyccs": response_only_ccs,
        "law_articles": law_articles,
        "similar_cases": similar_cases,
        "accusation": meta["accusation"],
        "articles": [f"刑法第{a}条" for a in meta["relevant_articles"]],
        "term": meta["term_of_imprisonment"]["imprisonment"],
        "punish_of_money": meta["punish_of_money"],
        "criminals": meta["criminals"]
    }
    results.append(result_item)

os.makedirs(os.path.dirname(output_json), exist_ok=True)
with open(output_json, "w", encoding="utf-8") as fout:
    json.dump(results, fout, ensure_ascii=False, indent=2)