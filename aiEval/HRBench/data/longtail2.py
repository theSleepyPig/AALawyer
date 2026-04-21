import json
import random
from collections import Counter, defaultdict

def load_data(file_path):
    """读取 JSON/JSONL 文件并返回包含原始数据的列表"""
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                data_list = data
        except json.JSONDecodeError:
            f.seek(0)
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data_list.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return data_list

def main():
    # 1. 配置文件路径 (请根据实际文件名修改)
    file_train = 'data_train.json'
    file_valid = 'data_valid.json'   # 如果你的验证集叫 data_val.json，请在这里修改
    output_file = 'longtail_fromtrain_dataset.json'
    
    # --- 核心超参数配置 ---
    # 定义长尾阈值：在训练集中出现次数 <= 该值的法条被定义为“长尾法条”
    LONG_TAIL_THRESHOLD = 50  
    # 采样数量：每个长尾法条在验证集中最多抽取几条？
    SAMPLES_PER_ARTICLE = 3   
    
    print("正在加载数据集 (可能需要十几秒)...")
    data_train = load_data(file_train)
    data_valid = load_data(file_valid)
    print(f"加载完毕！训练集: {len(data_train)} 条, 验证集: {len(data_valid)} 条")

    # 2. 统计训练集中的法条频率，圈定长尾法条
    train_article_counter = Counter()
    for item in data_train:
        if 'meta' in item and 'relevant_articles' in item['meta']:
            train_article_counter.update(item['meta']['relevant_articles'])
            
    # 筛选出频率小于等于阈值的法条
    tail_articles = {art for art, count in train_article_counter.items() if count <= LONG_TAIL_THRESHOLD}
    
    # 另外，那些在验证集里有，但在训练集里出现 0 次的 (Zero-shot)，也算作绝对的长尾
    valid_articles = set()
    for item in data_valid:
        if 'meta' in item and 'relevant_articles' in item['meta']:
            valid_articles.update(item['meta']['relevant_articles'])
            
    zero_shot_articles = valid_articles - set(train_article_counter.keys())
    tail_articles.update(zero_shot_articles)
    
    print(f"\n--- 训练集长尾统计 ---")
    print(f"训练集涉及法条总数: {len(train_article_counter)}")
    print(f"长尾法条 (<= {LONG_TAIL_THRESHOLD} 次) 数量: {len(tail_articles)} (包含 Zero-shot 法条 {len(zero_shot_articles)} 个)")

    # 3. 从验证集中定向捞取包含长尾法条的数据
    article_to_val_samples = defaultdict(list)
    for item in data_valid:
        if 'meta' in item and 'relevant_articles' in item['meta']:
            item_articles = set(item['meta']['relevant_articles'])
            # 找出这条数据中属于长尾的法条
            for article in item_articles.intersection(tail_articles):
                article_to_val_samples[article].append(item)

    print(f"在验证集中，成功找到了 {len(article_to_val_samples)} 种长尾法条的对应测试数据。")

    # 4. 针对找到的长尾法条进行均匀抽样
    long_tail_dataset = []
    selected_items_ids = set() # 用 fact 内容去重
    
    random.seed(42) # 固定随机种子以保证复现
    
    for article, available_samples in article_to_val_samples.items():
        if not available_samples:
            continue
            
        num_to_sample = min(SAMPLES_PER_ARTICLE, len(available_samples))
        sampled = random.sample(available_samples, num_to_sample)
        
        for item in sampled:
            fact_text = item.get('fact', '')
            if fact_text not in selected_items_ids:
                long_tail_dataset.append(item)
                selected_items_ids.add(fact_text)
                
    # 5. 保存生成的专属长尾测试集
    print(f"\n✅ 抽样完成！共抽取了 {len(long_tail_dataset)} 条数据。")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in long_tail_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"长尾测试集已成功保存至: {output_file}")

if __name__ == "__main__":
    main()