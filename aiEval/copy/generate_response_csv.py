import csv
import json
import re
import os
import torch
import transformers
import numpy as np
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer
from FlagEmbedding import FlagModel
from tqdm import tqdm 

# 路径设置
input_json = "/home/yxma/hzx/hzx/LeLLM/aiEval/aieval_dataset_100.json" 
output_csv = "results/generated_100.csv" 
model_path = "/mnt/ssd_2/yxma/LeLLM/train_mergem20"

# 模型加载
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transformers.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 法条数据库
with open("/mnt/ssd_2/yxma/LeLLM/data/data/RAGDatabase1.json", "r", encoding="utf-8") as file:
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
def generate_response(prompt):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=2048, do_sample=False)
    generated_ids = [out[len(inp):] for inp, out in zip(model_inputs.input_ids, generated_ids)]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

def extract_law_numbers(text):
    return re.findall(r"刑法第(\\d+)条", text)

def retrieve_law_articles(law_numbers):
    return "\n".join([f"刑法第{num}条: {law_data.get(num, '未找到相关法条')}" for num in law_numbers])

def retrieve_similar_cases(query, top_k=1):
    query_embedding = bge_model.encode([query])[0].astype("float32")
    D, I = faiss_index.search(np.array([query_embedding]), top_k)
    return [index_texts[int(index_ids[idx])] for idx in I[0]]

# 执行主流程
with open(input_json, "r", encoding="utf-8") as f_in:
    data = [json.loads(line) for line in f_in if line.strip()]


os.makedirs(os.path.dirname(output_csv), exist_ok=True)
with open(output_csv, "w", encoding="utf-8", newline="") as f_out:
    writer = csv.writer(f_out)
    writer.writerow([
        "input", "law_numbers", "response_analysis_no_rag",
        "response_analysis_with_case", "accusation", "articles",
        "term", "punish_of_money", "criminals"
    ])

    for idx, item in enumerate(tqdm(data, desc="Generating answers:")):
        user_input = item["fact"]
        meta = item["meta"]


        # 法条预测
        prompt_law = (
            "根据下列事实和罪名给出涉及的刑法法条。"
            "只需给出刑法法条编号，请将答案填在[法条]与<eoa>之间。\n"
            f"事实: {user_input}\n"
        )
        response_law_numbers = generate_response(prompt_law)
        law_numbers = extract_law_numbers(response_law_numbers)
        law_articles = retrieve_law_articles(law_numbers)

        # 无 RAG 分析
        prompt_no_rag = (
            f"案件内容: {user_input}\n"
            "根据以上案件内容，说明谁犯罪了，为什么认为他犯罪，涉及到哪条法律，犯了什么罪。结合案件内容和法条内容详细对比分析，把法条内容相和案件相关的整合进分析中。分析在100字左右。\n"
        )
        response_no_rag = generate_response(prompt_no_rag)

        # RAG 分析
        similar_cases = retrieve_similar_cases(user_input, top_k=1)
        prompt_rag = (
            f"案件内容: {user_input}\n"
            f"涉及法条内容:\n{law_articles}\n"
            "请根据案件内容、涉及的法条，以及如下提供的相似案例，综合分析案件。说明谁犯罪了，为什么认为他犯罪，涉及哪条法律，犯了什么罪。长度和格式参考相似案例。\n"
            f"【相似案例】:\n" + "\n\n".join(similar_cases) + "\n\n"
        )
        response_with_case = generate_response(prompt_rag)

        # 写入结果
        writer.writerow([
            user_input,
            "、".join([f"刑法第{n}条" for n in law_numbers]),
            response_no_rag,
            response_with_case,
            "、".join(meta["accusation"]),
            "、".join([f"刑法第{a}条" for a in meta["relevant_articles"]]),
            meta["term_of_imprisonment"]["imprisonment"],
            meta["punish_of_money"],
            "、".join(meta["criminals"])
        ])

        # print(f"✅ 第 {idx+1} 条处理完成")
