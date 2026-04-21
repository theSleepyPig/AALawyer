import re
import json
from pathlib import Path

# --- 配置输入和输出文件名 ---
INPUT_TEXT_FILE = '中华人民共和国刑法(2023修正).txt'
OUTPUT_JSON_FILE = 'RAGDatabase_latest_final.json' # 使用新文件名以避免混淆

# --- 中文数字到阿拉伯数字的转换逻辑 ---
CHINESE_NUMERALS_MAP = {
    '零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
    '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
    '十': 10, '百': 100
}

def convert_chinese_to_arabic(s: str) -> int:
    """
    将中文数字（例如 "一百二十三"）转换为阿拉伯数字（123）。
    针对刑法法条编号的特点进行了优化。
    """
    if not s or not isinstance(s, str):
        return 0
    if s.isdigit():
        return int(s)
        
    s = s.strip()
    total = 0
    
    # "十"的特殊处理
    if s.startswith('十'):
        s = '一' + s

    # 处理 "一百"、"二百" 等
    parts = s.split('百')
    if len(parts) > 1:
        hundreds_part = parts[0]
        if hundreds_part:
            total += convert_chinese_to_arabic(hundreds_part) * 100
        rest_part = parts[1]
    else:
        rest_part = s
        
    # 处理 "十" 及以下
    parts = rest_part.split('十')
    if len(parts) > 1:
        tens_part = parts[0]
        if tens_part:
            total += convert_chinese_to_arabic(tens_part) * 10
        else: # 处理 "十一" "十二"
            total += 10
        ones_part = parts[1]
        if ones_part:
            total += convert_chinese_to_arabic(ones_part)
    else:
        total += CHINESE_NUMERALS_MAP.get(rest_part, 0)
        
    # 如果转换失败（例如纯单位词），尝试简单映射
    if total == 0 and len(s) == 1 and s in CHINESE_NUMERALS_MAP:
        return CHINESE_NUMERALS_MAP[s]

    # 最后的简单转换逻辑，作为备用
    if total == 0:
        temp_val = 0
        for char in s:
            num = CHINESE_NUMERALS_MAP.get(char)
            if num is not None and num < 10:
                temp_val = temp_val * 10 + num
        return temp_val

    return total


def parse_article_key(article_title: str) -> str:
    """
    从法条标题中解析出用于JSON的key。
    例如："第一百二十条之一" -> "120-1"
           "第二百八十条" -> "280"
    """
    match = re.search(r'第(.+?)条', article_title)
    if not match:
        return None
    
    content = match.group(1)
    
    if '之' in content:
        parts = content.split('之')
        main_num_str = parts[0]
        sub_num_str = parts[1]
        
        main_num = convert_chinese_to_arabic(main_num_str)
        sub_num = convert_chinese_to_arabic(sub_num_str)
        return f"{main_num}-{sub_num}"
    else:
        main_num = convert_chinese_to_arabic(content)
        return str(main_num)

def process_law_text_to_json(input_path: Path, output_path: Path):
    """
    读取刑法文本文件，处理并生成JSON格式的法条库。
    """
    if not input_path.exists():
        print(f"错误：输入文件 '{input_path}' 不存在。请确保文件路径和名称正确。")
        return

    print(f"正在读取文件: {input_path}...")
    
    try:
        content = input_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return
    
    # ############## 关键修改处 ##############
    # 使用更灵活的正则表达式，不再要求“第X条”必须在行首
    # 使用正向先行断言 `(?=...)` 来保留分割符本身在每个块的开头
    article_chunks = re.split(r'(?=第[一二三四五六七八九十百千零]+条)', content)
    # ######################################

    law_database = {}
    
    # 第一个块是“第一条”之前的内容（如目录），应跳过
    if not article_chunks or len(article_chunks) <= 1:
        print("错误：无法在文件中分割出任何法条。请检查文件内容和格式。")
        return
        
    print(f"初步分割得到 {len(article_chunks) - 1} 个法条块，正在处理...")

    for chunk in article_chunks[1:]: # 从第二个块开始处理
        clean_chunk = chunk.strip()
        if not clean_chunk:
            continue

        first_line = clean_chunk.split('\n', 1)[0].strip()
        article_key = parse_article_key(first_line)

        if article_key:
            # 格式化value，与旧文件格式保持一致
            formatted_value = f"《中华人民共和国刑法》{clean_chunk}"
            law_database[article_key] = formatted_value
        else:
            # 修正后的逻辑基本不会触发此警告
            print(f"警告：无法从以下内容解析法条编号，已跳过:\n'{first_line}'\n")

    # 写入JSON文件
    print(f"处理完成，共解析 {len(law_database)} 个法条。正在写入到文件: {output_path}...")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(law_database, f, ensure_ascii=False, indent=4)
        print(f"成功！最新的法条库已生成在: {output_path}")
    except Exception as e:
        print(f"写入JSON文件时发生错误: {e}")


if __name__ == '__main__':
    try:
        current_dir = Path(__file__).parent
    except NameError:
        current_dir = Path.cwd()
        
    input_file_path = current_dir / INPUT_TEXT_FILE
    output_file_path = current_dir / OUTPUT_JSON_FILE
    
    process_law_text_to_json(input_file_path, output_file_path)