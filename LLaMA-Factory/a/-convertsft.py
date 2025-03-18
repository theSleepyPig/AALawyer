import json

# 输入原始 JSON 文件（包含多个案件）
input_file = "/home/yxma/hzx/hzx/LeLLM/ckpt/data/data_train.json"
# 输出 JSON 文件（转换后的格式）
output_file = "/home/yxma/hzx/hzx/LeLLM/ckpt/data/data_train_sft.json"

def process_case(case):
    """将单个案件数据转换为 LLaMAFactory SFT 格式"""
    instruction = "请根据以下案件事实，判断被告人的罪名、相关法条，并给出刑罚信息。"
    input_text = case.get("fact", "无案件事实描述")

    # 提取案件相关数据
    accusations = case["meta"].get("accusation", [])
    relevant_articles = case["meta"].get("relevant_articles", [])
    imprisonment = case["meta"].get("term_of_imprisonment", {})

    # 构造 output 字符串
    output_text = {
        "罪名": accusations if accusations else ["未知"],
        "适用法条": [f"《刑法》第{article}条" for article in relevant_articles],
        "刑罚": {
            "无期徒刑": imprisonment.get("life_imprisonment", False),
            "死刑": imprisonment.get("death_penalty", False),
            "有期徒刑": f"{imprisonment.get('imprisonment', 0)} 年",
            "罚款": f"{case['meta'].get('punish_of_money', 0)} 元"
        }
    }

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text
    }

def main():
    # 逐行读取 JSONL 格式
    processed_data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                case = json.loads(line.strip())  # 逐行解析 JSONL
                processed_data.append(process_case(case))
            except json.JSONDecodeError as e:
                print(f"解析失败: {e}")

    # 保存到 JSON 文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

    print(f"数据转换完成，共处理 {len(processed_data)} 条案件数据，已保存至 {output_file}")

if __name__ == "__main__":
    main()
