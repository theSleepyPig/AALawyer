# test_reranker_multithresh.py
import json
import argparse
import re
from tqdm import tqdm
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder, util

# --- 工具函数 ---
def extract_law_ids_from_answer(answer_text):
    if not answer_text:
        return []
    ids = re.findall(r'第(\d+)', answer_text)
    return sorted(list(set(ids)))

# --- 主逻辑 ---
def main():
    parser = argparse.ArgumentParser(description="Test Retrieve-then-Rerank with multiple thresholds")
    # ✅ 定义阈值范围参数
    parser.add_argument("--thresh_start", type=float, default=0.6, help="Start of the threshold range")
    parser.add_argument("--thresh_end", type=float, default=0.8, help="End of the threshold range")
    parser.add_argument("--thresh_step", type=float, default=0.01, help="Step for the threshold range")
    parser.add_argument("--retrieve_top_n", type=int, default=10, help="Number of candidates to retrieve before re-ranking")
    args = parser.parse_args()

    # --- 模型和数据路径配置 ---
    retriever_model_path = '/mnt/ssd_2/yxma/LeLLM/bge-m3'
    reranker_model_name = '/mnt/ssd_2/yxma/LeLLM/bge-reranker-v2-m3'
    test_file = "/home/yxma/hzx/LeLLM/LawBench/data/zero_shot/3-1.json"
    corpus_file = "/mnt/ssd_2/yxma/LeLLM/data/data/RAGDatabase1_v2.json"
    output_dir = "baseline_predictions_rerank_thresholds"
    os.makedirs(output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- 加载数据 ---
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print("Loading and processing law corpus...")
    with open(corpus_file, 'r', encoding='utf-8') as f:
        raw_law_data = json.load(f)
    law_corpus = [{"id": article_id, "content": content} for article_id, content in raw_law_data.items()]
    corpus_map = {doc['id']: doc['content'] for doc in law_corpus}
    print("Law corpus loaded.")

    # --- 初始化模型 ---
    print(f"Loading Retriever model from: {retriever_model_path}")
    retriever = SentenceTransformer(retriever_model_path, device=device)
    corpus_contents = [doc['content'] for doc in law_corpus]
    print("Encoding law corpus with Retriever model...")
    corpus_embeddings = retriever.encode(corpus_contents, convert_to_tensor=True, show_progress_bar=True, batch_size=8)
    print("Corpus encoding complete.")
    print(f"Loading Re-ranker model: {reranker_model_name}")
    reranker = CrossEncoder(reranker_model_name, device=device)
    print("Models loaded successfully.")

    # --- 耗时部分：一次性完成所有召回和重排打分 ---
    all_scored_results = []
    for item in tqdm(test_data, desc="Stage 1: Retrieving and Re-ranking all items"):
        # query = item["instruction"] + "\n" + item["question"]
        query = item["question"]
        ground_truth_ids = extract_law_ids_from_answer(item.get("answer", ""))
        
        # RETRIEVE
        query_embedding = retriever.encode(query, convert_to_tensor=True).to(device)
        retrieved_hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=args.retrieve_top_n)[0]
        
        # RE-RANK
        rerank_input_pairs = [[query, corpus_map[doc_id]] for doc_id in combined_ids]
        
        candidates = []
        if rerank_input_pairs:
            # 1. 先获取原始的、未归一化的分数 (logits)
            raw_scores = reranker.predict(rerank_input_pairs, show_progress_bar=False)
            
            # 2. 手动应用 Sigmoid 函数将其转换为 [0, 1] 区间的概率值
            #    需要先将 numpy array/list 转换为 torch.Tensor
            reranker_scores = torch.sigmoid(torch.Tensor(raw_scores)).tolist()
            
            # 后续逻辑不变
            for i in range(len(reranker_scores)):
                candidates.append({
                    "id": combined_ids[i],
                    "rerank_score": float(reranker_scores[i])
                })
            candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        all_scored_results.append({
            "query": item["question"],
            "ground_truth": ground_truth_ids,
            "reranked_candidates": candidates
        })
    print("Stage 1 complete. All items have been re-ranked.")

    # --- 快速部分：根据阈值范围，循环生成所有JSON文件 ---
    thresholds_to_test = np.arange(args.thresh_start, args.thresh_end + args.thresh_step, args.thresh_step)

    for threshold in tqdm(thresholds_to_test, desc="Stage 2: Generating JSON files for each threshold"):
        predictions = []
        for scored_item in all_scored_results:
            # 过滤分数高于阈值的候选者
            threshold_candidates = [
                candidate for candidate in scored_item['reranked_candidates']
                if candidate['rerank_score'] >= threshold
            ]
            predicted_ids = [candidate['id'] for candidate in threshold_candidates]
            
            predictions.append({
                "query": scored_item["query"],
                "ground_truth": scored_item["ground_truth"],
                "predicted": predicted_ids
            })

        # 保存当前阈值的结果文件
        filename = f"predictions_bge-m3_reranker_thresh{threshold:.2f}.json"
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"\n✅ All {len(thresholds_to_test)} prediction files have been generated in '{output_dir}'.")

if __name__ == "__main__":
    main()