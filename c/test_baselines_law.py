import json
import argparse
import re
from tqdm import tqdm
import os
import torch
import torch.nn.functional as F
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModel, AutoTokenizer
import jieba

# --- 工具函数 ---
# def extract_law_ids_from_answer(answer_text):
#     if not answer_text:
#         return []
#     ids = re.findall(r'第(\d+)', answer_text)
#     return sorted(list(set(ids)))
# --- 修改后的工具函数 ---

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
        tokenized_corpus = [list(jieba.cut(doc['content'])) for doc in corpus]
        self.corpus = corpus
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 Index built.")

    def retrieve(self, query, k=5):
        tokenized_query = list(jieba.cut(query))
        top_docs = self.bm25.get_top_n(tokenized_query, self.corpus, n=k)
        return [doc['id'] for doc in top_docs]

# --- BGE 检索器 ---
class BGERetriever:
    def __init__(self, corpus, model_name='/mnt/ssd_2/yxma/LeLLM/bge-m3'):
        print(f"Initializing BGE Retriever with model: {model_name}")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=self.device)
        self.corpus = corpus
        corpus_contents = [doc['content'] for doc in corpus]
        print("Encoding law corpus (BGE)...")
        self.corpus_embeddings = self.model.encode(
            corpus_contents, 
            batch_size=8, 
            convert_to_tensor=True, 
            show_progress_bar=True,
            normalize_embeddings=True
        )
        self.corpus_embeddings = self.corpus_embeddings.to(self.device)
        print("Corpus encoding complete.")

    def retrieve(self, query, k=10, score_threshold=None):
        query_embedding = self.model.encode(query, convert_to_tensor=True, normalize_embeddings=True).to(self.device)
        hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=k)[0]
        if score_threshold:
            hits = [hit for hit in hits if hit['score'] >= score_threshold]
        return [self.corpus[hit['corpus_id']]['id'] for hit in hits]

# --- SAILER 检索器 (新增) ---
class SAILERRetriever:
    def __init__(self, corpus, model_name='CSHaitao/SAILER_zh'):
        print(f"Initializing SAILER Retriever with model: {model_name}")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 使用原生 Transformers 加载
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        self.corpus = corpus
        corpus_contents = [doc['content'] for doc in corpus]
        
        print("Encoding law corpus (SAILER)...")
        self.corpus_embeddings = self.encode_batch(corpus_contents)
        print("Corpus encoding complete.")

    def encode_batch(self, texts, batch_size=16):
        """SAILER 专门的编码函数：使用 [CLS] Token 并归一化"""
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="SAILER Encoding"):
            batch_texts = texts[i : i + batch_size]
            encoded_input = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                # 取 [CLS] token (即第一个 token) 作为句子向量
                batch_embeddings = model_output.last_hidden_state[:, 0]
                # 归一化 (Dense Retrieval 标准操作)
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                
            all_embeddings.append(batch_embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0).to(self.device)

    def retrieve(self, query, k=10, score_threshold=None):
        # 编码 Query (同样取 CLS 并归一化)
        encoded_query = self.tokenizer(
            [query], 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            query_output = self.model(**encoded_query)
            query_embedding = query_output.last_hidden_state[:, 0]
            query_embedding = F.normalize(query_embedding, p=2, dim=1)

        # 使用 sentence_transformers 的工具函数计算余弦相似度 (因为都已经归一化了)
        hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=k)[0]
        
        if score_threshold:
            hits = [hit for hit in hits if hit['score'] >= score_threshold]
        
        return [self.corpus[hit['corpus_id']]['id'] for hit in hits]

# --- 主逻辑 ---
def main():
    parser = argparse.ArgumentParser(description="Test Retrieval Baselines for FAP task")
    # ✅ 1. 添加 sailer 到选项
    parser.add_argument("--method", type=str, required=True, choices=["bm25", "bge", "sailer"], help="Retrieval method")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top results to retrieve")
    parser.add_argument("--threshold", type=float, default=None, help="Score threshold (BGE/SAILER)")
    parser.add_argument("--model_path", type=str, default=None, help="Optional: Override model path")
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

    # ✅ 2. 初始化检索器
    if args.method == "bm25":
        retriever = BM25Retriever(law_corpus)
    elif args.method == "bge":
        path = args.model_path if args.model_path else '/mnt/ssd_2/yxma/LeLLM/bge-m3'
        retriever = BGERetriever(law_corpus, model_name=path)
    elif args.method == "sailer":
        # 默认使用 HuggingFace 上的 CSHaitao/SAILER_zh
        # 如果下载慢，可以先手动下载并指定 path
        path = args.model_path if args.model_path else '/mnt/ssd_2/yxma/LeLLM/SAILER_zh'
        retriever = SAILERRetriever(law_corpus, model_name=path)
    
    predictions = []
    desc_text = f"Running {args.method.upper()} baseline"
    for item in tqdm(test_data, desc=desc_text):
        query = item["question"]
        ground_truth_ids = extract_law_ids_from_answer(item.get("answer", ""))
        
        if args.threshold is not None and args.method in ['bge', 'sailer']:
            predicted_ids = retriever.retrieve(query, score_threshold=args.threshold, k=args.top_k)
        else:
            predicted_ids = retriever.retrieve(query, k=args.top_k)
        
        predictions.append({
            "query": item["question"],
            "ground_truth": ground_truth_ids,
            "predicted": predicted_ids
        })

    # --- 保存结果 ---
    output_dir = "baseline_predictions_bm_2"
    os.makedirs(output_dir, exist_ok=True)
    thresh_str = f"_thresh{args.threshold}" if args.threshold else ""
    filename = f"predictions_{args.method}_topk{args.top_k}{thresh_str}.json"
    output_path = os.path.join(output_dir, filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Baseline testing complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()