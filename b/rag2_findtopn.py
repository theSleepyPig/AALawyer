import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import numpy as np
import faiss
from FlagEmbedding import FlagModel

# ==== 配置 ====
index_path = "/mnt/ssd_2/yxma/LeLLM/data/RAG2/legalCase_faiss.index"
id_path = "/mnt/ssd_2/yxma/LeLLM/data/RAG2/legalCase_ids.json"
text_path = "/mnt/ssd_2/yxma/LeLLM/data/data/merge.json"
model_path = "/mnt/ssd_2/yxma/LeLLM/bge-large-zh"
top_k = 5

# ==== 加载模型 ====
model = FlagModel(model_path, 
                  query_instruction_for_retrieval="为这个句子生成表示：", 
                  device="cuda:0", 
                  use_multi_process=False)

# ==== 加载索引和数据 ====
index = faiss.read_index(index_path)
with open(id_path, "r", encoding="utf-8") as f:
    ids = json.load(f)
with open(text_path, "r", encoding="utf-8") as f:
    records = json.load(f)
texts = [r["text"].strip() for r in records]

# ==== 检索函数 ====
def search(query: str, top_k=2):
    query_embedding = model.encode([query])[0].astype("float32")
    D, I = index.search(np.array([query_embedding]), top_k)
    print(f"\n🔍 输入: {query}")
    for rank, idx in enumerate(I[0]):
        matched_text = texts[int(ids[idx])]
        print(f"\n🏷️ Top {rank+1} 相似文本:")
        # print(matched_text[:500] + ("..." if len(matched_text) > 500 else ""))
        print(matched_text)
        print(f"🧮 距离: {D[0][rank]:.4f}")
        

# ==== 示例 ====
search("近日，内蒙古自治区通辽铁路运输法院审理了苗某非法占用农用地罪案。公诉机关指控，苗某受利益驱使，将自家经营的草场进行翻扣种植葵花，经鉴定苗某摧毁天然牧草地面积300余亩，致使草原上原有植被严重毁坏。被告人苗某辩称:“案涉地块是集体组织分给我的，虽是草地，但草的长势并不好，近年来葵花籽价格较高，为了提高收入来源，我在自家草原上种植农作物，咋还犯罪了呢?我也是看见别人这么干，我才开垦草地的，我并不知道自己触犯了法律。")
