# test_baselines.py (最终版)
import json
import argparse
import re
from tqdm import tqdm
import os
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import jieba

# --- 工具函数 ---
# def extract_law_ids_from_answer(answer_text):
#     if not answer_text:
#         return []
#     ids = re.findall(r'第(\d+)', answer_text)
#     return sorted(list(set(ids)))

def extract_law_ids_from_answer(answer_text):
    """
    修复版：能够正确提取 "第23、24条" 或 "第23条，第24条" 中的所有数字
    """
    if not answer_text:
        return []
    
    ids = []
    
    # 策略：先提取 "第" 和 "条" 之间的内容（包含数字、顿号、逗号、空格）
    # 这里的正则解释：
    #  第        : 匹配字面量 "第"
    #  (         : 开始捕获组
    #    [\d\s、,，]+ : 匹配一个或多个：数字、空白、顿号、英文逗号、中文逗号
    #  )         : 结束捕获组
    #  条        : 匹配字面量 "条"
    matches = re.findall(r'第([\d\s、,，]+)条', answer_text)
    
    for match in matches:
        # match 可能是 "23、24" 或者 "102"
        # 在捕获的片段中，再次提取所有纯数字
        nums = re.findall(r'\d+', match)
        ids.extend(nums)
        
    # 如果列表为空（说明可能没有写"条"字，虽然LawBench很少见），
    # 或者是为了兼容旧逻辑，可以保留一个保底策略（可选）:
    if not ids:
         ids = re.findall(r'第(\d+)', answer_text)

    return sorted(list(set(ids)))

# --- BM25 检索器 ---
class BM25Retriever:
    def __init__(self, corpus):
        print("Initializing BM25 Retriever...")
        # ✅ 2. 使用 jieba.cut 进行中文分词
        tokenized_corpus = [list(jieba.cut(doc['content'])) for doc in corpus]
        self.corpus = corpus
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 Index built.")

    def retrieve(self, query, k=5):
        # ✅ 3. 对查询也使用 jieba.cut 进行分词
        tokenized_query = list(jieba.cut(query))
        top_docs = self.bm25.get_top_n(tokenized_query, self.corpus, n=k)
        return [doc['id'] for doc in top_docs]

# --- BGE 检索器 ---
class BGERetriever:
    # def __init__(self, corpus, model_name='/mnt/ssd_2/yxma/LeLLM/bge-m3'):
    def __init__(self, corpus, model_name="/mnt/ssd_2/yxma/LeLLM/bge-large-zh"):
        print(f"Initializing BGE Retriever with model: {model_name}")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.corpus = corpus
        corpus_contents = [doc['content'] for doc in corpus]
        print("Encoding law corpus...")
        self.corpus_embeddings = self.model.encode(
            corpus_contents, 
            batch_size=8, 
            convert_to_tensor=True, 
            show_progress_bar=True
        )
        self.corpus_embeddings = self.corpus_embeddings.to(self.device)
        print("Corpus encoding complete.")

    def retrieve(self, query, k=10, score_threshold=None): # 默认k可以设大一点，比如10，为阈值过滤提供更多候选
        query_embedding = self.model.encode(query, convert_to_tensor=True).to(self.device)

        # 1. 先进行一次标准的 Top-K 搜索，获取带分数的完整结果
        #    这里的 'hits' 是一个字典列表，每个字典包含 'corpus_id' 和 'score'
        hits = util.semantic_search(
            query_embedding,
            self.corpus_embeddings,
            top_k=k
        )[0]

        # 2. 如果设置了阈值 (score_threshold is not None)，就手动过滤
        if score_threshold:
            # 列表推导式：只保留那些 'score' 大于等于阈值的结果
            hits = [hit for hit in hits if hit['score'] >= score_threshold]
        
        # 3. 返回最终结果的 ID
        return [self.corpus[hit['corpus_id']]['id'] for hit in hits]

# --- 主逻辑 ---
def main():
    parser = argparse.ArgumentParser(description="Test Retrieval Baselines for FAP task")
    parser.add_argument("--method", type=str, required=True, choices=["bm25", "bge"], help="Retrieval method to use")
    # ✅ 新增参数
    parser.add_argument("--top_k", type=int, default=5, help="Number of top results to retrieve for Top-K mode")
    parser.add_argument("--threshold", type=float, default=None, help="Score threshold for Threshold mode (only for BGE)")
    args = parser.parse_args()

    # --- 加载数据 ---
    test_file = "/home/yxma/hzx/LeLLM/LawBench/data/zero_shot/3-1.json" 
    corpus_file = "/mnt/ssd_2/yxma/LeLLM/data/data/RAGDatabase1_v2.json"

    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print("Loading and processing law corpus...")
    with open(corpus_file, 'r', encoding='utf-8') as f:
        raw_law_data = json.load(f)
    law_corpus = [{"id": article_id, "content": content} for article_id, content in raw_law_data.items()]
    print("Law corpus loaded.")

    # 初始化检索器
    if args.method == "bm25":
        retriever = BM25Retriever(law_corpus)
    elif args.method == "bge":
        retriever = BGERetriever(law_corpus)
    
    predictions = []
    for item in tqdm(test_data, desc=f"Running {args.method.upper()} baseline"):
        # query = item["instruction"] + "\n" + item["question"]
        query = item["question"]
        ground_truth_ids = extract_law_ids_from_answer(item.get("answer", ""))
        
        # ✅ 根据模式调用检索器
        if args.threshold is not None and args.method == 'bge':
            predicted_ids = retriever.retrieve(query, score_threshold=args.threshold, k=args.top_k)
        else:
            predicted_ids = retriever.retrieve(query, k=args.top_k)
        
        predictions.append({
            "query": item["question"],
            "ground_truth": ground_truth_ids,
            "predicted": predicted_ids
        })

    # --- 保存预测结果，文件名包含参数信息 ---
    # output_dir = "baseline_predictions1_woin"
    output_dir = "baseline_predictions_v1.5_2"
    os.makedirs(output_dir, exist_ok=True)
    if args.threshold is not None and args.method == 'bge':
        filename = f"predictions_{args.method}_thresh{args.threshold}.json"
    else:
        filename = f"predictions_{args.method}_topk{args.top_k}.json"
    output_path = os.path.join(output_dir, filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Baseline testing complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()