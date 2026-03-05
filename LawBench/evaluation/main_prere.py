# main.py (修改后)
import json
import os
import pandas as pd
# ✅ 确保您自己的 ljp_article.py 文件在可导入的路径
from evaluation_functions import ljp_article_v2
import sys
import argparse

def read_json(input_file):
    # load the json file
    with open(input_file, "r", encoding="utf-8") as f:
        data_dict = json.load(f)

    # 您的数据已经是list of dicts格式，无需转换
    # 如果您的 prediction.json 是 dict of dicts 格式，请取消下面的注释
    dict_size = len(data_dict)
    new_data_dict = []
    for i in range(dict_size):
        example = data_dict[str(i)]
        new_data_dict.append(example)
    return new_data_dict
    
    # return data_dict


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", dest="input_folder",
                  help="input folder: it should be a folder containing prediction results for task 3-1", metavar="FILE")
    parser.add_argument("-o", "--outfile", dest="outfile",
                  help="output file saving the evaluation results", metavar="FILE")
    args = parser.parse_args(argv)

    # ✅ 我们只关心 3-1 任务
    target_task = "3-1"
    score_function = ljp_article_v2.compute_ljp_article

    input_dir = args.input_folder
    output_file = args.outfile
    
    # ✅ 增加新的列来保存 precision 和 recall
    results = {
        "model_name": [], 
        "f1_score": [], 
        "precision": [], 
        "recall": [],
        "abstention_rate": []
    }
    
    # list all folders in input_dir
    system_folders = os.listdir(input_dir)
    for system_folder in system_folders:
        if system_folder.startswith("."):
            continue
        
        system_folder_dir = os.path.join(input_dir, system_folder)
        if not os.path.isdir(system_folder_dir):
            continue
        
        print(f"*** Evaluating System: {system_folder} ***")
        
        # ✅ 直接寻找 3-1.json 文件
        input_file = os.path.join(system_folder_dir, f"{target_task}.json")

        if not os.path.exists(input_file):
            print(f"*** Warning: {input_file} not found, skipping. ***")
            continue

        data_dict = read_json(input_file)
        
        print(f"Processing {target_task}:")
        score_dict = score_function(data_dict) # 返回的是一个字典
        print(f"Scores for {target_task}: {score_dict}")

        # ✅ 从返回的字典中获取所有分数
        results["model_name"].append(system_folder)
        results["f1_score"].append(score_dict.get("f1_score", 0))
        results["precision"].append(score_dict.get("precision", 0))
        results["recall"].append(score_dict.get("recall", 0))
        abstention_rate = score_dict.get("abstention_rate", 0)
        results["abstention_rate"].append(abstention_rate)

        print(f"*** Evaluating System: {system_folder} Done! ***")
        print()

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False, float_format='%.5f')
    print(f"✅ Evaluation complete. Full report saved to: {output_file}")

if __name__ == '__main__':
    main(sys.argv[1:])