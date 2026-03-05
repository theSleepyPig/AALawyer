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

## 2️⃣ Main Training

The training process of our AA-LeLLM backbone consists of Incremental Pretraining (IPT) and Supervised Fine-Tuning (SFT). 

### Stage 1: Incremental Pretraining (IPT)
```bash
# [TODO: Add your IPT training command here]
```

### Stage 2: Supervised Fine-Tuning (SFT) & SCR Training
```bash
# [TODO: Add your multi-task SFT command here].
```

## 3️⃣ Main Evaluation

We evaluate our model on LawBench and our proposed Hallucination Risk-Benchmark (HR-Benchmark).

### Evaluate on LawBench (FAP, CP, DFI, etc.)
```bash
# [TODO: Add LawBench evaluation script command]
```

### Evaluate on HR-Benchmark (Hallu, Prof, Info, Expa)
```bash
# [TODO: Add HR-Benchmark evaluation script command using LLM-as-a-Judge]deepseek-chat
```

## 4️⃣ Running AALawyer (Inference)

Input (-> SCR -> APR) -> Final Legal Analysis:

```bash
# [TODO: Add inference/demo command here]
```

## 5️⃣ ⭐ Pretrained Models & Datasets

Our fine-tuned models and newly collected datasets are available on Hugging Face 🤗:

- **Model Weights (AA-LeLLM):** <font color="red">TODO</font>[TODO: Add Hugging Face Link]
- **APR Criminal Case Database (176k cases):** <font color="red">TODO</font>[TODO: Add Hugging Face Link]


## 6️⃣ Citation

If you find this work helpful, please consider citing us:<font color="red">TODO</font>

```bibtex
coming soon
```

<!-- 0️⃣ 1️⃣ 2️⃣ 3️⃣ 4️⃣ 5️⃣ 6️⃣ 7️⃣ 8️⃣ 9️⃣ 🔟
1️⃣1️⃣ 1️⃣2️⃣ 1️⃣3️⃣ 1️⃣4️⃣ 1️⃣5️⃣ 1️⃣6️⃣ 1️⃣7️⃣ 1️⃣8️⃣ 1️⃣9️⃣ 2️⃣0️⃣ ⭐ 🌟 -->