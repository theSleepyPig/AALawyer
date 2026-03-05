import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("/mnt/ssd_2/yxma/LeLLM/Fuzi-Mingcha-6B/", trust_remote_code=True)
model = AutoModel.from_pretrained("/mnt/ssd_2/yxma/LeLLM/Fuzi-Mingcha-6B/", trust_remote_code=True).half().cuda()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)