import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
# python LLaMA-Factory/a/chat.py --model_path /mnt/ssd_2/yxma/LeLLM/train_mergem20

parser = argparse.ArgumentParser(description="Interactive Chat with LLM")
parser.add_argument("--model_path", type=str, required=True, help="Path to the local model")
parser.add_argument("--device", type=str, default="cuda:3", help="CUDA devices to use (comma-separated)")
parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum new tokens to generate")
args = parser.parse_args()

# # 解析设备列表
# device_list = [int(d) for d in args.device.split(",")]
# device_map_setting = {str(i): device_list[i] for i in range(len(device_list))} if len(device_list) > 1 else {"": device_list[0]}
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

print("Welcome! Type your question below (type 'exit' to quit).")

def generate_response(prompt):
    """
    使用本地 LLM 生成回复
    """
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda:3")

    generated_ids = model.generate(
        **model_inputs, 
        max_new_tokens=args.max_new_tokens
        # repetition_penalty=2.0
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
    
    response = generate_response(user_input)
    print(f"AA-LawLLM: {response}")
