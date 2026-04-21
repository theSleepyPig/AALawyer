import re
import json
from pathlib import Path

# --- 配置輸入和輸出文件名 ---
INPUT_TEXT_FILE = '中华人民共和国刑法(2023修正).txt'
OUTPUT_JSON_FILE = 'RAGDatabase_FINAL_PERFECT.json' # 使用一個全新的、清晰的文件名

# --- 中文數字到阿拉伯數字的轉換邏輯 (已徹底重寫並嚴格測試) ---
def convert_chinese_to_arabic(s: str) -> int:
    """
    一個完全重寫的、健壯的中文數字轉換函數，用於正確處理法律條文編號。
    """
    if not s or not isinstance(s, str): return 0
    if s.isdigit(): return int(s)
    
    s = s.strip()
    
    num_map = {'零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
    unit_map = {'十': 10, '百': 100}
    
    res = 0
    temp_num = 0 # 臨時存放單位前的數字，如 "二" in "二百"
    
    # 遍歷字符串中的每個字符
    for char in s:
        if char in num_map:
            temp_num = num_map[char]
        elif char in unit_map:
            unit = unit_map[char]
            # 如果單位前沒有數字，默認為1 (例如 "十" = 10, "百" = 100)
            if temp_num == 0 and unit == 10:
                temp_num = 1
            res += temp_num * unit
            temp_num = 0
    
    # 將最後的個位數加上
    res += temp_num
    
    return res

def parse_article_key(article_header: str) -> (str, str):
    """
    從法條標題中精確解析出主鍵和副鍵。
    """
    match = re.search(r'第([一二三四五六七八九十百零]+)条(?:之([一二三四五六七八九十百零]+))?', article_header)
    if not match:
        return None, None
    
    main_num_str, sub_num_str = match.groups()
    
    main_key = str(convert_chinese_to_arabic(main_num_str)) if main_num_str else None
    sub_key = str(convert_chinese_to_arabic(sub_num_str)) if sub_num_str else None
    
    return main_key, sub_key

def process_law_text_to_json(input_path: Path, output_path: Path):
    """
    讀取刑法文本文件，精確處理並生成JSON法條庫。
    """
    if not input_path.exists():
        print(f"錯誤：輸入文件 '{input_path}' 不存在。")
        return

    print(f"正在讀取文件: {input_path}...")
    try:
        content = input_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"讀取文件時發生錯誤: {e}")
        return

    # 這個定位邏輯是正確的，它能找到所有法條的邊界
    matches = list(re.finditer(r'^\s*第[一二三四五六七八九十百零]+条(?:之[一二三四五六七八九十百零]+)?.*$', content, re.MULTILINE))
    
    if not matches:
        print("錯誤：無法在文件中解析出任何法條。請檢查文件格式。")
        return

    law_database = {}
    print(f"識別到 {len(matches)} 個獨立的法條條目，正在進行解析與合併...")

    for i, match in enumerate(matches):
        start_pos = match.start()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        chunk = content[start_pos:end_pos].strip()
        title_line = chunk.split('\n', 1)[0].strip()
        
        main_key, sub_key = parse_article_key(title_line)

        if not main_key or main_key == '0':
            print(f"警告：無法正確解析法條編號，跳過: '{title_line}'")
            continue

        formatted_chunk = f"《中华人民共和国刑法》{chunk}"

        if sub_key:
            if main_key in law_database:
                law_database[main_key] += f"\n\n{formatted_chunk}"
            else:
                print(f"警告：找到了'之'條款但主法條'{main_key}'不存在，已獨立創建。")
                law_database[f"{main_key}-{sub_key}"] = formatted_chunk
        else:
            law_database[main_key] = formatted_chunk
            
    print(f"處理完成，共生成 {len(law_database)} 個獨立的法條編號。正在寫入到文件: {output_path}...")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(law_database, f, ensure_ascii=False, indent=4)
        print(f"成功！最終完美的法條庫已生成在: {output_path}")
    except Exception as e:
        print(f"寫入JSON文件時發生錯誤: {e}")

if __name__ == '__main__':
    try:
        script_dir = Path(__file__).parent
    except NameError:
        script_dir = Path.cwd()

    input_file_path = script_dir / INPUT_TEXT_FILE
    output_file_path = script_dir / OUTPUT_JSON_FILE
    
    process_law_text_to_json(input_file_path, output_file_path)