# Mitigating Legal Hallucinations via Symbolic Constraints and Analogical Precedents

A complementary dual-retriever framework (AALawyer) designed to mitigate legal hallucinations using Symbolic Constrained Retrieval (SCR) and Analogical Precedent Retrieval (APR).

⭐ Our paper is available at <font color="red">TODO</font>[TODO: Link to be added].

## 1️⃣ Environment

Please ensure you have the following dependencies installed:

- python: 3.9.21
- pyTorch: 2.6.0
- cuda: 12.4
- transformers: 4.45.2
- LLaMA-Factory: 0.9.2
- sentence-transformers: 4.0.1
- FlagEmbedding: 1.3.4
- peft: 0.12.0

```bash
conda env create -f env.yml
conda activate llm
```

> **Note on `transformers` version:**> The `transformers` library frequently updates its underlying architectures. We used `4.45.2` for our specific models (DeepSeek-7B). If you plan to experiment with different versions or families of base models, you may need to manually adjust (upgrade or downgrade) the `transformers` version to match their specific architectural requirements to avoid compatibility errors.

## 2️⃣ Incremental Pretraining and Supervised Fine-Tuning

The training process of our AA-LeLLM backbone consists of Incremental Pretraining (IPT) and Supervised Fine-Tuning (SFT). 

### Merging LoRA Weights
Since we use LoRA, the trained adapters must be merged back into the base model before proceeding to the next stage. **You must perform this merge operation after each of the 3 IPT steps and before the SFT stage.**

<details>
<summary>🖱️ <code>llamafactory-cli export ...</code> <b>[Click to Expand Merge Command Template]</b></summary>

We have provided a ready-to-use configuration file for merging. Please click and navigate to [`LLaMA-Factory/a/merge.yaml`](LLaMA-Factory/a/merge.yaml) in our repository to modify the `model_name_or_path`, `adapter_name_or_path`, and `export_dir` to match your current training step.

Once the paths are updated, run the following command:

```bash
# Modify the paths accordingly for base_model and adapter to get tuned model.
cd LLaMA-Factory
llamafactory-cli export a/merge.yaml
```
*(Note: Repeat this merge process for `train_m11` to `train_mergem11` and `train_m12` to `train_mergem12` respectively before starting Stage 2.)*
</details>

### Stage 1: IPT
<details>
<summary>🖱️ <code>llamafactory-cli train --stage pt ...</code> <b>[Click to Expand Full Command]</b></summary>

```bash
cd LLaMA-Factory
# =================================================================
# Step 1: Incremental Pretraining on all articles (Generate m10)
# =================================================================
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train \
    --stage pt \
    --do_train True \
    --model_name_or_path [The_path_of_the_base_model(e.g.DeepSeek-R1-Distill-Qwen-7B)] \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template deepseek3 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset [dataset_name(e.g.articles-all)] \
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
    --output_dir [xxx(e.g.train_m10)] \
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

# =================================================================
# Step 2: Incremental Pretraining on legal instruments (Generate m11)
# =================================================================
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train \
    --stage pt \
    --do_train True \
    --model_name_or_path [The_path_of_the_base_model(e.g.train_mergem10)] \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template deepseek3 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset [dataset_name(e.g.legal-instrument-all)] \
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
    --output_dir [xxx(e.g.train_m11)] \
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

# =================================================================
# Step 3: Incremental Pretraining on Criminal law articles (Generate m12)
# =================================================================
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train \
    --stage pt \
    --do_train True \
    --model_name_or_path [The_path_of_the_base_model(e.g.train_mergem11)] \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template deepseek3 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset [dataset_name(e.g.articles-only-penal)] \
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
    --output_dir [xxx(e.g.train_m12)] \
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
```
</details>

### Stage 2: SFT (including SCR Training)
<details>
<summary>🖱️ <code>llamafactory-cli train --stage sft ...</code> <b>[Click to Expand Full Command]</b></summary>

```bash
cd LLaMA-Factory
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path [The_path_of_the_base_model(e.g.train_mergem12)] \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template deepseek3 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset [dataset_name(e.g.accusations)] \
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
    --output_dir [xxx(e.g.train_m20)] \
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
```
</details>

## 3️⃣ Main Evaluation

We evaluate our model on LawBench and our proposed Hallucination Risk-Benchmark (HR-Benchmark).

### Evaluate on LawBench (FAP, CP, DFI, etc.)
```bash

```

### Evaluate on RAG (F1 Score of FAP)
```bash
cd c
bash run_all_thresholds.sh
python analyze_final_results2.py
```
Before running the evaluation, you may need to adjust the following files to match your file and model paths:
* [`test_baselines_v15.py`](./c/test_baselines_v15.py): The core retrieval script that performs legal article searches by top-$k$ and threshold. This is also the primary location for switching or extending retrieval models (e.g., by adding a new Retriever class).
* [`run_all_thresholds.sh`](./c/run_all_thresholds.sh): A shell script that automates the execution of the retrieval script across a range of thresholds for ablation studies.
* [`analyze_final_results2.py`](./c/analyze_final_results2.py): A post-processing script that evaluates above results, calculates metrics, and identifies the optimal threshold ($\tau$).
### Evaluate on HR-Benchmark (Hallu, Prof, Info, Expa)
```bash

```

## 4️⃣ Running AALawyer (Interactive Web UI)

We provide an interactive web interface for users to experience the full AALawyer pipeline: **User Input → Symbolic Constrained Retrieval (SCR) + Analogical Precedent Retrieval (APR) → Final Legal Analysis**.

To launch the local Web UI, simply run the following command:

```bash
cd Interface
python app.py --model_path [model_path(e.g.train_mergem20)]
```

## 5️⃣ Ablations


## 5️⃣ ⭐ Pretrained Models & Datasets

Our fine-tuned models and newly collected datasets are available on Hugging Face 🤗:

- **Model Weights (AA-LeLLM):** <font color="red">TODO</font>[TODO: Add Hugging Face Link]
- **APR Criminal Case Database (176k cases):** <font color="red">TODO</font>[TODO: Add Hugging Face Link]


## 6️⃣ Citation

If you find this work helpful, please consider citing us:<font color="red">TODO</font>

```bibtex
coming soon
```

## 7️⃣ Acknowledgements

We would like to express our sincere gratitude to the following open-source projects:

- **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)**: For providing an excellent training framework that  supported the fine-tuning process of our work.
- **[LawBench](https://github.com/open-compass/LawBench)**: For providing a comprehensive evaluation benchmark for Legal Large Language Models.

**Code Declaration:** Please note that this repository includes and builds upon source code from both LLaMA-Factory and LawBench. We have made specific modifications and improvements to their original codebases to seamlessly integrate them with our AALawyer framework and evaluation pipeline. We thank the original authors for their phenomenal contributions to the community.

<!-- 0️⃣ 1️⃣ 2️⃣ 3️⃣ 4️⃣ 5️⃣ 6️⃣ 7️⃣ 8️⃣ 9️⃣ 🔟
1️⃣1️⃣ 1️⃣2️⃣ 1️⃣3️⃣ 1️⃣4️⃣ 1️⃣5️⃣ 1️⃣6️⃣ 1️⃣7️⃣ 1️⃣8️⃣ 1️⃣9️⃣ 2️⃣0️⃣ ⭐ 🌟 -->