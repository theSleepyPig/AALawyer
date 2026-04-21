import json
import random
from collections import defaultdict

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

def extract_articles(data_list):
    """提取数据集中所有的法条集合"""
    articles = set()
    for item in data_list:
        if 'meta' in item and 'relevant_articles' in item['meta']:
            articles.update(item['meta']['relevant_articles'])
    return articles

def main():
    file_full = 'data_valid.json'
    file_200 = 'aieval_dataset_200.json'
    output_file = 'longtail_dataset.json'
    
    # 抽取数量：每个长尾法条抽取多少个样本？(建议2-3个即可)
    SAMPLES_PER_ARTICLE = 2 

    print("正在加载数据集...")
    data_full = load_data(file_full)
    data_200 = load_data(file_200)
    
    # 1. 找出 200 条子集中没出现过的长尾法条
    articles_full = extract_articles(data_full)
    articles_200 = extract_articles(data_200)
    
    missing_articles = articles_full - articles_200
    print(f"完整集包含 {len(articles_full)} 种法条。")
    print(f"评测子集(200条)包含 {len(articles_200)} 种法条。")
    print(f"发现未覆盖的的长尾法条共 {len(missing_articles)} 种！")
    
    # 2. 建立 长尾法条 -> 包含该法条的数据列表 的映射字典
    article_to_samples = defaultdict(list)
    for item in data_full:
        if 'meta' in item and 'relevant_articles' in item['meta']:
            # 找出这条数据包含的长尾法条
            item_articles = set(item['meta']['relevant_articles'])
            for article in item_articles.intersection(missing_articles):
                article_to_samples[article].append(item)
                
    # 3. 针对每个长尾法条进行随机抽样
    long_tail_dataset = []
    # 使用 set 去重，因为一条数据可能同时包含多个长尾法条
    selected_items_ids = set() 
    
    random.seed(42) # 固定随机种子，保证每次生成的测试集一样
    
    for article in missing_articles:
        available_samples = article_to_samples[article]
        if not available_samples:
            continue
            
        # 如果该法条的总样本数小于我们要抽的数量，就全拿；否则随机抽指定数量
        num_to_sample = min(SAMPLES_PER_ARTICLE, len(available_samples))
        sampled = random.sample(available_samples, num_to_sample)
        
        for item in sampled:
            # 简单的基于内容去重（防止同一条文本被抽取多次）
            fact_text = item.get('fact', '')
            if fact_text not in selected_items_ids:
                long_tail_dataset.append(item)
                selected_items_ids.add(fact_text)
                
    # 4. 保存生成的长尾测试集
    print(f"\n抽样完成！共抽取了 {len(long_tail_dataset)} 条数据作为长尾测试集。")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # 按照 JSONL 格式写入（每行一个JSON），和原始数据保持一致
        for item in long_tail_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"长尾测试集已成功保存至: {output_file}")

if __name__ == "__main__":
    main()
    
#根据200分布中val没有测试到的