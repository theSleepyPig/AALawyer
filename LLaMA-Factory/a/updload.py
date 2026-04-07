# export HF_HOME=/home/yxma/hzx/huggingface_cache
from huggingface_hub import login, upload_folder


# login()


upload_folder(folder_path="/mnt/ssd_2/yxma/LeLLM/", repo_id="xuanzhu07/AALawyer_Data", repo_type="dataset")