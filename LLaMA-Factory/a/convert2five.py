import json

# 输入 JSONL 文件路径
input_file = "/home/yxma/hzx/hzx/LeLLM/ckpt/data/data_train.json"

# 输出文件路径
output_files = {
    "法条": "/home/yxma/hzx/hzx/LeLLM/ckpt/data/data_train_sft/articles.json",
    "罪名": "/home/yxma/hzx/hzx/LeLLM/ckpt/data/data_train_sft/accusations.json",
    "罚金": "/home/yxma/hzx/hzx/LeLLM/ckpt/data/data_train_sft/fines.json",
    "犯罪人": "/home/yxma/hzx/hzx/LeLLM/ckpt/data/data_train_sft/criminals.json",
    "刑期": "/home/yxma/hzx/hzx/LeLLM/ckpt/data/data_train_sft/sentences.json"
}

# 任务模板
TEMPLATES = {
    "法条": {
        "instruction": "根法法条。只需给出刑法法条编据下列事实和罪名给出涉及的刑号，请将答案填在[法条]与<eoa>之间。例如[法条]刑法第128条、刑法第341条<eoa>",
        "question": "事实:{fact}\n罪名:{accusation}",
        "answer": "法条:[法条]{articles}<eoa>"
    },
    "罪名": {
        "instruction": "请你模拟法官依据下面事实给出罪名，只需要给出罪名的名称，将答案写在[罪名]和<eoa>之间。例如[罪名]盗窃;诈骗<eoa>。请你严格按照这个格式回答。",
        "question": "事实:{fact}",
        "answer": "罪名:[罪名]{accusation}<eoa>"
    },
    "罚金": {
        "instruction": "请依据下列事实和罪名，判断罚金金额。只需给出罚金金额，请将答案填在[罚金]与<eoa>之间。例如[罚金]5000元<eoa>。",
        "question": "事实:{fact}\n罪名:{accusation}",
        "answer": "罚金:[罚金]{fine}元<eoa>"
    },
    "犯罪人": {
        "instruction": "依据给出的实体类型提取句子的实体信息，实体类型包括:犯罪嫌疑人。逐个列出实体信息。请将答案填在[犯罪嫌疑人]与<eoa>之间。例如[犯罪嫌疑人]张三<eoa>。",
        "question": "事实:{fact}",
        "answer": "犯罪嫌疑人:[犯罪嫌疑人]{criminals}<eoa>"
    },
    "刑期": {
        "instruction": "根据下列事实、罪名和刑法法条预测判决刑期。只需给出判决刑期为多少月，请将答案填在[刑期]与<eoa>之间。例如[刑期]12月<eoa>。",
        "question": "事实:{fact}\n罪名:{accusation}\n法条:{articles}",
        "answer": "刑期:[刑期]{sentence}个月<eoa>"
    }
}

def process_case(case):
    """转换单个案件数据到5种不同格式"""
    fact = case.get("fact", "无案件事实描述")
    accusations = "; ".join(case["meta"].get("accusation", []))  # 罪名
    articles = "、".join([f"刑法第{article}条" for article in case["meta"].get("relevant_articles", [])])  # 法条
    fine = case["meta"].get("punish_of_money", 0)  # 罚金
    criminals = "; ".join(case["meta"].get("criminals", []))  # 犯罪人
    imprisonment = case["meta"].get("term_of_imprisonment", {})  # 刑期
    sentence = imprisonment.get("imprisonment", 0) * 12  # 以月计算刑期

    return {
        "法条": {
            "instruction": TEMPLATES["法条"]["instruction"],
            "input": TEMPLATES["法条"]["question"].format(fact=fact, accusation=accusations),
            "output": TEMPLATES["法条"]["answer"].format(articles=articles)
        },
        "罪名": {
            "instruction": TEMPLATES["罪名"]["instruction"],
            "input": TEMPLATES["罪名"]["question"].format(fact=fact),
            "output": TEMPLATES["罪名"]["answer"].format(accusation=accusations)
        },
        "罚金": {
            "instruction": TEMPLATES["罚金"]["instruction"],
            "input": TEMPLATES["罚金"]["question"].format(fact=fact, accusation=accusations),
            "output": TEMPLATES["罚金"]["answer"].format(fine=fine)
        },
        "犯罪人": {
            "instruction": TEMPLATES["犯罪人"]["instruction"],
            "input": TEMPLATES["犯罪人"]["question"].format(fact=fact),
            "output": TEMPLATES["犯罪人"]["answer"].format(criminals=criminals)
        },
        "刑期": {
            "instruction": TEMPLATES["刑期"]["instruction"],
            "input": TEMPLATES["刑期"]["question"].format(fact=fact, accusation=accusations, articles=articles),
            "output": TEMPLATES["刑期"]["answer"].format(sentence=sentence)
        }
    }

def main():
    # 逐行读取 JSONL 数据
    dataset = {key: [] for key in output_files}  # 初始化数据存储

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                case = json.loads(line.strip())  # 逐行解析 JSONL
                case_data = process_case(case)  # 处理数据
                for key in dataset:
                    dataset[key].append(case_data[key])  # 存储不同任务的数据
            except json.JSONDecodeError as e:
                print(f"解析失败: {e}")

    # 分别写入 5 个 JSON 文件
    for key, file_path in output_files.items():
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(dataset[key], f, ensure_ascii=False, indent=4)

    print(f"数据转换完成，共处理 {len(dataset['法条'])} 条案件数据，已保存至 5 个 JSON 文件！")

if __name__ == "__main__":
    main()
