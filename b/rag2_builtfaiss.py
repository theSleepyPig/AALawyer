import os
import json
import time
import numpy as np
import faiss
from tqdm import tqdm
from FlagEmbedding import FlagModel

def main():
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # ==== 配置 ====
    json_path = "/mnt/ssd_2/yxma/LeLLM/data/data/merge.json"
    save_index_path = "/mnt/ssd_2/yxma/LeLLM/data/RAG2/legalCase_faiss.index"
    save_id_path = "/mnt/ssd_2/yxma/LeLLM/data/RAG2/legalCase_ids.json"
    save_failed_path = "/mnt/ssd_2/yxma/LeLLM/data/RAG2/legalCase_failed_ids.json"
    model_path = "/mnt/ssd_2/yxma/LeLLM/bge-large-zh"
    batch_size = 64

    print("加载模型中...")
    model = FlagModel(model_path, 
                  query_instruction_for_retrieval="为这个句子生成表示：", 
                  use_fp16=True,
                  device="cuda:0",
                  use_multi_process=False)  # 禁用并行

    print(f"加载数据文件：{json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    texts = [r["text"].strip() for r in records]
    ids = [str(i) for i in range(len(texts))]

    print(f"共 {len(texts):,} 条文本数据，开始向量化...")
    start_time = time.time()

    all_embeddings = []
    all_ids = []
    failed_ids = []

    for i in tqdm(range(0, len(texts), batch_size), desc="向量化中", ncols=100):
        batch_texts = texts[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        try:
            # emb = model.encode(batch_texts, is_query=False)
            emb = model.encode(batch_texts)
            all_embeddings.append(np.array(emb))
            all_ids.extend(batch_ids)
        except Exception as e:
            print(f"❌ 第{i}-{i+batch_size}条向量化失败：{e}")
            failed_ids.extend(batch_ids)

    if not all_embeddings:
        print("❌ 没有成功向量化的内容，终止构建索引。")
        return

    embeddings = np.vstack(all_embeddings).astype("float32")
    dimension = embeddings.shape[1]
    print(f"✅ 成功向量化条数：{len(all_ids)}")
    print(f"❌ 向量化失败条数：{len(failed_ids)}")
    print(f"✅ 向量维度: {dimension}, 总样本数: {embeddings.shape[0]:,}")

    print("🔧 构建 FAISS IndexFlatL2 索引...")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs(os.path.dirname(save_index_path), exist_ok=True)
    faiss.write_index(index, save_index_path)

    with open(save_id_path, "w", encoding="utf-8") as f:
        json.dump(all_ids, f, ensure_ascii=False, indent=2)

    with open(save_failed_path, "w", encoding="utf-8") as f:
        json.dump(failed_ids, f, ensure_ascii=False, indent=2)

    print(f"\n✅ FAISS 向量库构建完成！")
    print(f"索引路径: {save_index_path}")
    print(f"成功ID路径: {save_id_path}")
    print(f"失败ID路径: {save_failed_path}")
    print(f"总耗时：{time.time() - start_time:.2f} 秒")

if __name__ == "__main__":
    main()
