import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
model_path = "/mnt/ssd_2/yxma/LeLLM/DISC-LawLLM"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained(model_path)
messages = []
messages.append({"role": "user", "content": "生产销售假冒伪劣商品罪如何判刑？"})
response = model.chat(tokenizer, messages)
print(response)
