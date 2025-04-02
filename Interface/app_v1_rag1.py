import json
import re
import torch
import argparse
from flask import Flask, request, render_template, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, MarianMTModel, MarianTokenizer
import transformers

# python app.py --model_path /mnt/ssd_2/yxma/LeLLM/train_mergem20

# 解析参数
parser = argparse.ArgumentParser(description="Interactive Chat with LLM")
parser.add_argument("--model_path", type=str, required=True, help="Path to the local model")
parser.add_argument("--device", type=str, default="cuda:3", help="CUDA devices to use (comma-separated)")
parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum new tokens to generate")
args = parser.parse_args()

# 选择设备
device = torch.device(args.device if torch.cuda.is_available() else "cpu")

transformers.logging.set_verbosity_error()

print("Loading model on GPUs...")
model = AutoModelForCausalLM.from_pretrained(
    args.model_path, 
    torch_dtype=torch.float16, 
    device_map=device  
)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

# 加载翻译模型
print("Loading translation models...")
zh_to_en_model = MarianMTModel.from_pretrained("/mnt/ssd_2/yxma/LeLLM/opus-mt-zh-en").to(device)
zh_to_en_tokenizer = MarianTokenizer.from_pretrained("/mnt/ssd_2/yxma/LeLLM/opus-mt-zh-en")

# 读取法条数据库
law_file = "/mnt/ssd_2/yxma/LeLLM/data/data/RAGDatabase1.json"
with open(law_file, "r", encoding="utf-8") as file:
    law_data = json.load(file)

print("Law articles loaded successfully!")

# 创建 Flask 应用
app = Flask(__name__)

def extract_law_numbers(text):
    """ 从 LLM 生成的文本中提取多个法条编号 """
    return re.findall(r"刑法第(\d+)条", text)

def retrieve_law_articles(law_numbers):
    """ 根据多个法条编号从 JSON 文件中检索对应的法条内容 """
    articles = [f"刑法第{num}条: {law_data.get(num, '未找到相关法条')}" for num in law_numbers]
    return "\n".join(articles) if articles else "未找到相关法条"

def translate(text, model, tokenizer):
    """ 翻译文本 """
    translated = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    translated_tokens = model.generate(**translated, max_length=512)
    # return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    translated_texts = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    return " ".join(translated_texts)

def generate_response(prompt):
    """ 使用 LLM 生成回复 """
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        **model_inputs, 
        max_new_tokens=args.max_new_tokens
    )
    
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()
    # 只获取 LLM 生成的部分，不包括输入的 Prompt
    # response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # # **清理掉输入的 Prompt**
    # clean_response = response_text.replace(prompt, "").strip()
    
    # return clean_response

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form["case_description"]

        # 1. 生成 LLM 预测的法条编号
        prompt_law = (
            "根据下列事实和罪名给出涉及的刑法法条。"
            "只需给出刑法法条编号，请将答案填在[法条]与<eoa>之间。\n"
            "例如[法条]刑法第128条、刑法第341条<eoa>\n"
            f"事实: {user_input}\n"
        )

        response_law_numbers = generate_response(prompt_law)
        law_numbers = extract_law_numbers(response_law_numbers)

        # 2. 查询法条内容
        law_articles = retrieve_law_articles(law_numbers)

        # 3. 生成 LLM 分析案件
        prompt_analysis = (
            # "根据下列案件内容和相关法条分析案件，说明谁犯罪了，为什么认为他犯罪，涉及到哪条法律，犯了什么罪。分析在100字左右。分析时不需要提及法条内容，只提及相应编号。\n"
            
            f"案件内容: {user_input}\n"
            f"涉及法条内容:\n{law_articles}\n"
            "根据以上案件内容和相关法条分析案件，说明谁犯罪了，为什么认为他犯罪，涉及到哪条法律，犯了什么罪。结合案件内容和法条内容详细对比分析，把法条内容相和案件相关的整合进分析中。分析在100字左右。\n"
            
        )

        response_analysis = generate_response(prompt_analysis)

        # 4. 翻译 LLM 生成的分析
        translated_articles = translate(law_articles, zh_to_en_model, zh_to_en_tokenizer)
        translated_analysis = translate(response_analysis, zh_to_en_model, zh_to_en_tokenizer)

        return render_template("index.html", case=user_input, prediction=response_law_numbers, laws=law_articles, analysis=response_analysis, translated_analysis=translated_analysis, translated_articles = translated_articles)

    return render_template("index.html", case="", prediction="", laws="", analysis="", translated_analysis="", translated_articles = "")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
