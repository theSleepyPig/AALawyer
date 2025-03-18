import json
from tqdm import tqdm  # 引入 tqdm 进度条库
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 读取数据
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# 保存处理后的数据
def save_data(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def compute_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    cosine_sim = cosine_similarity(vectorizer[0:1], vectorizer[1:2])
    return cosine_sim[0][0]

# 判断是否重复
def is_duplicate(test_entry, train_data, threshold=0.5):
    test_fact = test_entry.get('instruction', '').split("事实:")[-1].strip()
    if not test_fact:
        print(f"Skipping entry due to empty or missing 'testinstruction': {test_entry}")  
        return False  

    for train_entry in train_data:
        train_fact = train_entry.get('instruction', '').split("事实:")[-1].strip()
        if not train_fact:
            continue  
        similarity = compute_cosine_similarity(test_fact, train_fact)
        if similarity > threshold:  
            return True  # 找到重复
    return False  # 没有找到重复


def remove_duplicates(test_data, train_data, threshold=0.5):
    filtered_test_data = []
    for idx, test_entry in enumerate(tqdm(test_data, desc="Processing test data", unit="entry")):
        if 'instruction' not in test_entry:
            print(f"❌ Missing 'traininstruction' in entry: {test_entry}")  
        else:
            duplicate = is_duplicate(test_entry, train_data, threshold)
            if duplicate:
                print(f"❌ Duplicate found for test entry {idx}, remove")
            else:
                print(f"✅ No duplicate for test entry {idx}")
                filtered_test_data.append(test_entry)  
    return filtered_test_data

# 主函数
def main():
    train_file_path = '/mnt/ssd_2/yxma/LeLLM/data/data/data_train_sft.json'
    test_file_path = '/home/yxma/hzx/hzx/LeLLM/LawBench/data/zero_shot/3-1.json'
    output_file_path = '/home/yxma/hzx/hzx/LeLLM/LawBench/data/zero_shot/3-1_filtered.json'  

    print("Loading data...")
    train_data = load_data(train_file_path)
    test_data = load_data(test_file_path)

    print("Removing duplicates...")
    filtered_test_data = remove_duplicates(test_data, train_data)

    print("Saving filtered data...")
    save_data(filtered_test_data, output_file_path)
    print(f"✅ 去重后的数据已保存到: {output_file_path}")

if __name__ == '__main__':
    main()
# nohup python /home/yxma/hzx/hzx/LeLLM/LLaMA-Factory/a/cleantest.py > output1.log 2>&1 &