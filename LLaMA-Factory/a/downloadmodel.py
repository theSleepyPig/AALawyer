from huggingface_hub import snapshot_download

# 下载 Qwen2.5-0.5B
# snapshot_download(repo_id="Qwen/Qwen2.5-0.5B", local_dir="./qwen2.5-0.5B")
# model_path = "/mnt/ssd_2/yxma/LeLLM/Qwen2.5-Math-7B"
# snapshot_download(repo_id="Qwen/Qwen2.5-0.5B-Instruct", local_dir="/mnt/ssd_2/yxma/LeLLM/qwen2.5-0.5B-instruct")
# snapshot_download(repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", local_dir="/mnt/ssd_2/yxma/LeLLM/DeepSeek-R1-Distill-Qwen-7B")
# snapshot_download(repo_id="Qwen/Qwen2.5-Math-7B", local_dir="/mnt/ssd_2/yxma/LeLLM/Qwen2.5-Math-7B")
# 下载 Fuzi-Mingcha 6B 模型
# snapshot_download(repo_id="SDUIRLab/fuzi-mingcha-v1_0", local_dir="/mnt/ssd_2/yxma/LeLLM/Fuzi-Mingcha-6B")

# snapshot_download(
#     repo_id="ShengbinYue/LawLLM-7B", 
#     local_dir="/mnt/ssd_2/yxma/LeLLM/LawLLM-7B"
# )

snapshot_download(
    repo_id="CSHaitao/SAILER_zh", 
    local_dir="/mnt/ssd_2/yxma/LeLLM/SAILER_zh"
)