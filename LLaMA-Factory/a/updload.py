# export HF_HOME=/home/yxma/hzx/huggingface_cache
from huggingface_hub import login, upload_folder, upload_file


# login()


# upload_folder(folder_path="/mnt/ssd_2/yxma/LeLLM/data/data", repo_id="xuanzhu07/APR_Criminal_Case_Database", repo_type="dataset")
upload_file(
    path_or_fileobj="/mnt/ssd_2/yxma/hzx/data/data/merge.json", # "/mnt/ssd_2/yxma/hzx/data/data/merge.json"
    path_in_repo="merge.json",                                    # 上传到 Hugging Face 后显示的文件名
    repo_id="xuanzhu07/APR_Criminal_Case_Database",
    repo_type="dataset"
)