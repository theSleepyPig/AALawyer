import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, MarianMTModel, MarianTokenizer
import transformers

# python LLaMA-Factory/a/chaten.py --model_path /mnt/ssd_2/yxma/LeLLM/train_mergem20
# python a/chaten.py --model_path /mnt/ssd_2/yxma/LeLLM/train_mergem20

parser = argparse.ArgumentParser(description="Interactive Chat with LLM")
parser.add_argument("--model_path", type=str, required=True, help="Path to the local model")
parser.add_argument("--device", type=str, default="cuda:3", help="CUDA devices to use (comma-separated)")
parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum new tokens to generate")
args = parser.parse_args()

# 选择设备（单 GPU）
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
# en_to_zh_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-zh").to(device)
zh_to_en_model = MarianMTModel.from_pretrained("/mnt/ssd_2/yxma/LeLLM/opus-mt-zh-en").to(device)
# en_to_zh_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
zh_to_en_tokenizer = MarianTokenizer.from_pretrained("/mnt/ssd_2/yxma/LeLLM/opus-mt-zh-en")

print("Welcome! Type your question below (type 'exit' to quit).")

def translate(text, model, tokenizer):
    """
    Translate text using the specified model and tokenizer
    """
    translated = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    translated_tokens = model.generate(**translated, max_length=512)
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text

def split_text(text, max_len=512):
    """
    Split long text into smaller parts to avoid token length exceeding model's limit
    """
    tokens = tokenizer.encode(text)
    return [tokens[i:i+max_len] for i in range(0, len(tokens), max_len)]

def generate_response(prompt):
    """
    使用本地 LLM 生成回复
    """
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

# 进入交互模式
while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        print("Exiting chat...")
        break

    # 1. 获取输入的中文并翻译成英文
    translated_input = ""
    chunks = split_text(user_input)  # 拆分长文本
    for chunk in chunks:
        part = tokenizer.decode(chunk)
        translated_part = translate(part, zh_to_en_model, zh_to_en_tokenizer)
        translated_input += translated_part
    print(f"Translated to English: {translated_input}")
    print()
    
    # 2. 使用 LLM 模型生成中文回复
    prompt_analysis = (
            "根据下列案件内容分析案件，说明谁犯罪了，为什么认为他犯罪，涉及到哪条法律，犯了什么罪。结合案件内容和法条内容详细对比分析，分析在100字左右。\n"
            f"案件内容: {user_input}\n"
            
        )

    response_zh = generate_response(prompt_analysis)
    print(f"AA-LawLLM : {response_zh}")
    # print()
    
    # 3. 翻译输出中文回复成英文
    translated_output = ""
    chunks = split_text(response_zh)  # 拆分生成的中文回复
    for chunk in chunks:
        part = tokenizer.decode(chunk)
        translated_part = translate(part, zh_to_en_model, zh_to_en_tokenizer)
        translated_output += translated_part
    print(f"Translated Response (English): {translated_output}")
    print()