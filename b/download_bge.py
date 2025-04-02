from transformers import AutoModel, AutoTokenizer

save_path = "/mnt/ssd_2/yxma/LeLLM/bge-large-zh"  # ← 改成你要保存的目录

model = AutoModel.from_pretrained("BAAI/bge-large-zh")
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-zh")

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"✅ 模型保存完成，路径：{save_path}")