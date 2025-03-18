CUDA_VISIBLE_DEVICES=0,1,2,3 GRADIO_SHARE=1 GRADIO_SERVER_PORT=7860 llamafactory-cli webui
nohup CUDA_VISIBLE_DEVICES=0,1,2,3 GRADIO_SHARE=1 GRADIO_SERVER_PORT=7860 llamafactory-cli webui > outputsft.log 2>&1 &



/mnt/ssd_2/yxma/LeLLM/qwen2.5-0.5B-instruct
/mnt/ssd_2/yxma/LeLLM/DeepSeek-R1-Distill-Qwen-7B

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1


llamafactory-cli train \
    --stage sft \
    --model_name_or_path /mnt/ssd_2/yxma/LeLLM/qwen2.5-0.5B-instruct \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --quantization_method bitsandbytes \
    --template qwen \
    --flash_attn auto \
    --dataset_dir data \
    --eval_dataset identity \
    --cutoff_len 1024 \
    --max_samples 100000 \
    --per_device_eval_batch_size 2 \
    --predict_with_generate True \
    --max_new_tokens 512 \
    --top_p 0.7 \
    --temperature 0.95 \
    --output_dir saves/Qwen2.5-0.5B-Instruct/lora/eval_2 \
    --trust_remote_code True \
    --do_predict True 

# 预训练 lora + article --> m10
llamafactory-cli train \
    --stage pt \
    --do_train True \
    --model_name_or_path /mnt/ssd_2/yxma/LeLLM/DeepSeek-R1-Distill-Qwen-7B \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template deepseek3 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset articles-all \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 1.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing True \
    --report_to wandb \
    --run_name train_m10 \
    --output_dir /mnt/ssd_2/yxma/LeLLM/ckpt/train_m10 \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all 

# 预训练 lora + 文书 --> m11
llamafactory-cli train \
    --stage pt \
    --do_train True \
    --model_name_or_path /mnt/ssd_2/yxma/LeLLM/train_mergem10 \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template deepseek3 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset legal-instrument-all \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 1.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing True \
    --report_to wandb \
    --run_name train_m11 \
    --output_dir /mnt/ssd_2/yxma/LeLLM/ckpt/train_m11 \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all

# 预训练 lora + 刑法法条 --> m12
CUDA_VISIBLE_DEVICES=1,2,3 llamafactory-cli train \
    --stage pt \
    --do_train True \
    --model_name_or_path /mnt/ssd_2/yxma/LeLLM/train_mergem11 \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template deepseek3 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset articles-only-penal \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing True \
    --report_to wandb \
    --run_name train_m12 \
    --output_dir /mnt/ssd_2/yxma/LeLLM/ckpt/train_m12 \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all



# sft + 所有任务训练数据15w*5
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /mnt/ssd_2/yxma/LeLLM/train_mergem12 \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template deepseek3 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset accusations \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 1.0 \
    --max_samples 200000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir /mnt/ssd_2/yxma/LeLLM/ckpt/train_m20 \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all

1

llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /mnt/ssd_2/yxma/LeLLM/train_mergem12 \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template deepseek3 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset five_sft_tasks \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 1.0 \
    --max_samples 2000000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --report_to wandb \
    --run_name train_m20 \
    --output_dir /mnt/ssd_2/yxma/LeLLM/ckpt/train_m20 \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all
3
nohup CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /mnt/ssd_2/yxma/LeLLM/train_mergem12 \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template deepseek3 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset five_sft_tasks \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 1.0 \
    --max_samples 2000000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --report_to wandb \
    --run_name train_m20 \
    --output_dir /mnt/ssd_2/yxma/LeLLM/ckpt/train_m20 \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all > train.log 2>&1 &

5
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /mnt/ssd_2/yxma/LeLLM/train_mergem13 \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template deepseek3 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset nine_sft_tasks \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 1.0 \
    --max_samples 2000000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 2000 \
    --warmup_steps 0 \
    --packing False \
    --report_to wandb \
    --run_name train_m21 \
    --output_dir /mnt/ssd_2/yxma/LeLLM/ckpt/train_m21 \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all
# m21
nohup bash -c "CUDA_VISIBLE_DEVICES=0,1,2 llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /mnt/ssd_2/yxma/LeLLM/train_mergem13 \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template deepseek3 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset nine_sft_tasks \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 1.0 \
    --max_samples 2000000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 2000 \
    --warmup_steps 0 \
    --packing False \
    --report_to wandb \
    --run_name train_m20 \
    --output_dir /mnt/ssd_2/yxma/LeLLM/ckpt/train_m21 \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all" > train_m21.log 2>&1 &

# m22
nohup bash -c "CUDA_VISIBLE_DEVICES=0,1,2 llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /mnt/ssd_2/yxma/LeLLM/train_mergem12 \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template deepseek3 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset sftm22 \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 1.0 \
    --max_samples 2000000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 5000 \
    --warmup_steps 0 \
    --packing False \
    --report_to wandb \
    --run_name train_m22 \
    --output_dir /mnt/ssd_2/yxma/LeLLM/ckpt/train_m22 \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all" > train_m22.log 2>&1 &