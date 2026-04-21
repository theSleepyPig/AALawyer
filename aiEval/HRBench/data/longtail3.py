import json
import random
import re
from collections import Counter, defaultdict

def load_train_data(file_path):
    """读取训练集 JSONL 文件 (包含 meta.relevant_articles)"""
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # 尝试按行读取 JSONL
        for line in f:
            line = line.strip()
            if line:
                try:
                    data_list.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return data_list

def load_rag_test_data(file_path):
    """读取 3-1.json 格式的评测文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_articles_from_answer(answer_text):
    """利用正则表达式从 '法条:刑法第xxx条' 中提取法条数字列表"""
    # 匹配 '第' 和 '条' 之间的数字，例如 '第264条' -> 264
    matches = re.findall(r'第(\d+)条', answer_text)
    return [int(m) for m in matches]

def main():
    # 1. 配置文件路径
    file_train = 'data_train.json'
    file_test = 'lawbench_dataset.json'
    output_file = 'longtail_fromtrain_dataset.json'
    
    # --- 核心超参数配置 ---
    LONG_TAIL_THRESHOLD = 154   # 训练集中出现 <= 50 次的认定为长尾
    SAMPLES_PER_ARTICLE = 3    # 每个长尾法条在 3-1.json 中最多抽几条？(如果想要全部保留，可以设大点比如 100)
    
    print("正在加载数据集...")
    data_train = load_train_data(file_train)
    data_test = load_rag_test_data(file_test)
    print(f"加载完毕！训练集: {len(data_train)} 条, 评测集 (3-1.json): {len(data_test)} 条")

    # 2. 统计训练集中的法条频率
    train_article_counter = Counter()
    for item in data_train:
        if 'meta' in item and 'relevant_articles' in item['meta']:
            train_article_counter.update(item['meta']['relevant_articles'])
            
    # 圈定长尾法条集合
    tail_articles = {art for art, count in train_article_counter.items() if count <= LONG_TAIL_THRESHOLD}
    
    # 3. 遍历 3-1.json，寻找长尾数据并分类
    article_to_test_samples = defaultdict(list)
    test_articles_set = set()
    
    for item in data_test:
        answer_text = item.get('answer', '')
        # 解析文本中的法条编号
        item_articles = set(extract_articles_from_answer(answer_text))
        test_articles_set.update(item_articles)
        
        # 寻找既存在于 3-1.json 中，又属于长尾的法条 (包含 Zero-shot)
        for article in item_articles:
            # 如果在训练集中没出现过(Zero-shot) 或者 出现次数小于阈值
            if article not in train_article_counter or article in tail_articles:
                article_to_test_samples[article].append(item)

    # 统计 Zero-shot
    zero_shot_articles = test_articles_set - set(train_article_counter.keys())

    print(f"\n--- 统计信息 ---")
    print(f"3-1.json 包含的法条种类: {len(test_articles_set)}")
    print(f"发现的 RAG 长尾法条种类: {len(article_to_test_samples)} (含 Zero-shot 法条 {len(zero_shot_articles)} 种)")

    # 4. 针对长尾法条进行抽样构建最终的长尾评测集
    long_tail_dataset = []
    selected_questions = set() # 利用 question 文本去重
    
    random.seed(42) # 固定随机种子
    
    for article, available_samples in article_to_test_samples.items():
        if not available_samples:
            continue
            
        num_to_sample = min(SAMPLES_PER_ARTICLE, len(available_samples))
        sampled = random.sample(available_samples, num_to_sample)
        
        for item in sampled:
            question_text = item.get('question', '')
            if question_text not in selected_questions:
                long_tail_dataset.append(item)
                selected_questions.add(question_text)
                
    # 5. 保存生成的专属长尾评测集
    print(f"\n✅ 抽样完成！共抽取了 {len(long_tail_dataset)} 条 RAG 长尾数据。")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # 因为 3-1.json 是完整的 JSON 列表格式，我们按照原格式输出
        json.dump(long_tail_dataset, f, ensure_ascii=False, indent=2)
            
    print(f"长尾测试集已成功保存至: {output_file}")

if __name__ == "__main__":
    main()