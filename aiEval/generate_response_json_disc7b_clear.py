import csv
import json
import re
import os
import gc
import torch
import transformers
import numpy as np
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer
from FlagEmbedding import FlagModel
from tqdm import tqdm

# 路径设置
input_json = "/home/yxma/hzx/hzx/LeLLM/aiEval/aieval_dataset_200.json"
output_json = "results/generated_200_disc2.json"

model_path = "/mnt/ssd_2/yxma/LeLLM/LawLLM-7B"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transformers.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 法条数据库和 Faiss 检索系统
with open("/mnt/ssd_2/yxma/LeLLM/data/data/RAGDatabase1_v2.json", "r", encoding="utf-8") as file:
    law_data = json.load(file)
bge_model = FlagModel("/mnt/ssd_2/yxma/LeLLM/bge-large-zh", query_instruction_for_retrieval="为这个句子生成表示：", device="cuda:0", use_multi_process=False)
faiss_index = faiss.read_index("/mnt/ssd_2/yxma/LeLLM/data/RAG2/legalCase_faiss.index")
with open("/mnt/ssd_2/yxma/LeLLM/data/RAG2/legalCase_ids.json", "r", encoding="utf-8") as f:
    index_ids = json.load(f)
with open("/mnt/ssd_2/yxma/LeLLM/data/data/merge.json", "r", encoding="utf-8") as f:
    index_texts = [r["text"].strip() for r in json.load(f)]

# 基础函数
def extract_law_numbers(text):
    return re.findall(r"刑法第(\d+)条", text)

def retrieve_law_articles(law_numbers):
    return "\n".join([law_data.get(num, "未找到相关法条") for num in law_numbers])

def retrieve_similar_cases(query, top_k=1):
    query_embedding = bge_model.encode([query])[0].astype("float32")
    D, I = faiss_index.search(np.array([query_embedding]), top_k)
    return [index_texts[int(index_ids[idx])] for idx in I[0]]

def generate_response(prompt):
    full_prompt = f"你是LawLLM，一个由复旦大学DISC实验室创造的法律助手。\n用户：{prompt}\n助手："
    with torch.no_grad():
        model_inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True).to(device)
        generated_ids = model.generate(**model_inputs, max_new_tokens=1024, do_sample=False)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    del model_inputs, generated_ids
    torch.cuda.empty_cache()
    return response.strip()

# 加载数据
with open(input_json, "r", encoding="utf-8") as f_in:
    data = [json.loads(line) for line in f_in if line.strip()]

os.makedirs(os.path.dirname(output_json), exist_ok=True)
with open(output_json, "w", encoding="utf-8") as fout:
    fout.write("[\n")

    for idx, item in enumerate(tqdm(data, desc="Generating answers:")):
        user_input = item["fact"]
        meta = item["meta"]

        prompt_law = (
            "根据下列事实和罪名给出涉及的刑法法条。"
            "只需给出刑法法条编号，请将答案填在[法条]与<eoa>之间。\n"
            "例如[法条]刑法第128条、刑法第341条<eoa>\n"
            f"事实: {user_input}\n"
        )
        response_law_numbers = generate_response(prompt_law)
        law_numbers = extract_law_numbers(response_law_numbers)
        law_articles = retrieve_law_articles(law_numbers)

        prompt_no_rag = (
            f"案件内容: {user_input}\n"
            "根据以上案件内容，说明谁犯罪了，为什么认为他犯罪，涉及到哪条法律，犯了什么罪。"
            "结合案件内容和法条内容详细对比分析，把法条内容相和案件相关的整合进分析中。分析在100字左右。\n"
        )
        response_no_rag = generate_response(prompt_no_rag)

        prompt_aa = (
            f"案件内容: {user_input}\n"
            f"涉及法条内容:\n{law_articles}\n"
            "根据以上案件内容和相关法条分析案件，说明谁犯罪了，为什么认为他犯罪，涉及到哪条法律，犯了什么罪。"
            "结合案件内容和法条内容详细对比分析，把法条内容相和案件相关的整合进分析中。分析在100字左右。\n"
        )
        response_aa = generate_response(prompt_aa)

        similar_cases = retrieve_similar_cases(user_input, top_k=1)
        prompt_rag = (
            f"案件内容: {user_input}\n"
            f"涉及法条内容:\n{law_articles}\n"
            f"【相似案例】:\n" + "\n\n".join(similar_cases) + "\n\n"
            "请根据案件内容、涉及的法条，以及如下提供的相似案例，综合分析案件。"
            "说明谁犯罪了，为什么认为他犯罪，涉及哪条法律，犯了什么罪。长度和格式参考相似案例。\n"
        )
        response_with_case = generate_response(prompt_rag)

        prompt_ccs = (
            f"案件内容: {user_input}\n"
            f"【相似案例】:\n" + "\n\n".join(similar_cases) + "\n\n"
            "请根据案件内容、涉及的法条，以及如下提供的相似案例，综合分析案件。"
            "说明谁犯罪了，为什么认为他犯罪，涉及哪条法律，犯了什么罪。长度和格式参考相似案例。\n"
        )
        response_only_ccs = generate_response(prompt_ccs)

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

        # 分段写入 JSON 文件
        json.dump(result_item, fout, ensure_ascii=False, indent=2)
        fout.write(",\n" if idx != len(data) - 1 else "\n")

        # 清理显存与内存
        del result_item, response_no_rag, response_aa, response_with_case, response_only_ccs, similar_cases
        gc.collect()
        torch.cuda.empty_cache()

    fout.write("]\n")
