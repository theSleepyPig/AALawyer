import json
import os
import pandas as pd
from evaluation_functions import authen_simple, acc, jec_kd, cjft, ydlj, ftcs, jdzy, jetq, ljp_accusation, ljp_article, ljp_imprison, wbfl, xxcq, flzx, wsjd, yqzy, lblj, zxfl, sjjc
import sys
import argparse

def read_json(input_file):
    # load the json file
    with open(input_file, "r", encoding="utf-8") as f:
        data_dict = json.load(f)

    dict_size = len(data_dict)
    new_data_dict = []
    for i in range(dict_size):
        example = data_dict[str(i)]
        new_data_dict.append(example)

    return new_data_dict


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", dest="input_folder",
                  help="input folder: it should be a folder containing the prediction results", metavar="FILE")
    parser.add_argument("-o", "--outfile", dest="outfile",
                  help="output file saving the evaluation results", metavar="FILE")
    args = parser.parse_args(argv)
    funct_dict = {"4-1": acc.compute_ljp_article,
        "4-2": authen_simple.evaluate_prediction_vs_full_law
                # "3-6": jec_ac.compute_jec_ac,
                #   "1-2": jec_kd.compute_jec_kd,
                #   "3-2": cjft.compute_cjft,
                #   "3-8": flzx.compute_flzx,
                #   "1-1": ftcs.compute_ftcs,
                #   "2-2": jdzy.compute_jdzy,
                #   "3-7": jetq.compute_jetq,
                #   "3-3": ljp_accusation.compute_ljp_accusation,
                #   "3-1": ljp_article.compute_ljp_article,
}
                #   "3-4": ljp_imprison.compute_ljp_imprison,
                #   "3-5": ljp_imprison.compute_ljp_imprison,
                #   "2-3": wbfl.compute_wbfl,
                #   "2-6": xxcq.compute_xxcq,
                #   "2-1": wsjd.compute_wsjd,
                #   "2-4": zxfl.compute_zxfl,
                #   "2-7": yqzy.compute_yqzy,
                #   "2-8": lblj.compute_lblj,
                #   "2-5": ydlj.compute_ydlj,
                #   "2-9": sjjc.compute_sjjc,
                #   "2-10": sjjc.compute_cfcy}
    input_dir = args.input_folder

    output_file = args.outfile
    results = {"task": [], "model_name": [], "score": [], "abstention_rate": []}
    # list all folders in input_dir
    system_folders = os.listdir(input_dir)
    for system_folder in system_folders:
        if system_folder.startswith("."):
            continue
        system_folder_dir = os.path.join(input_dir, system_folder)
        if not os.path.isdir(system_folder_dir):
            continue
        # list all files in system_folder_path
        # dataset_files = os.listdir(system_folder_dir)
        dataset_files = sorted(os.listdir(system_folder_dir), key=lambda x: (x != "4-1.json", x))

        print(f"*** Evaluating System: {system_folder} ***")
        acc_result = None
        auth_result = None
        for dataset_file in dataset_files:
            datafile_name = dataset_file.split(".")[0]
            input_file = os.path.join(system_folder_dir, dataset_file)
            data_dict = read_json(input_file)
                
            if datafile_name == "4-2":
                # auth_result = score
                # ✅ auth0 from 4-1-test.json
                auth_result0 = None
                auth_result1 = None
                print("Using law_articles from 4-1-150.json to compute auth0")
                with open("/home/yxma/hzx/LeLLM/ckpt/predictions/hall/m0/4-1-150-v2023-m0.json", "r", encoding="utf-8") as f:
                    auth_data_0 = json.load(f)
                law_articles_list_0 = [{"prediction": item["law_articles"]} for item in auth_data_0]
                auth_result0 = authen_simple.evaluate_prediction_vs_full_law(law_articles_list_0)

                hall_result0 = authen_simple.compute_hall_score(
                    data_dict,
                    acc_result['acc_list'],
                    auth_result0['auth_list']
                )
                print(f"[hall0] Hallucination score = {hall_result0['score']:.4f}")
                results["task"].append("hall0")
                results["model_name"].append(system_folder)
                results["score"].append(hall_result0["score"])
                results["abstention_rate"].append(0)
                                                   
                results["task"].append("auth0")
                results["model_name"].append(system_folder)
                results["score"].append(auth_result0["score"])
                results["abstention_rate"].append(auth_result0.get("abstention_rate", 0))

                # ✅ auth1 from normal 4-2.json
                normal_4_2_path = os.path.join(system_folder_dir, "4-2.json")
                if os.path.exists(normal_4_2_path):
                    print("Using original 4-2.json to compute auth1")
                    normal_4_2_data = read_json(normal_4_2_path)
                    auth_result1 = authen_simple.evaluate_prediction_vs_full_law(normal_4_2_data)

                    hall_result1 = authen_simple.compute_hall_score(
                        data_dict,
                        acc_result['acc_list'],
                        auth_result1['auth_list']
                    )
                    print(f"[hall1] Hallucination score = {hall_result1['score']:.4f}")
                    results["task"].append("hall1")
                    results["model_name"].append(system_folder)
                    results["score"].append(hall_result1["score"])
                    results["abstention_rate"].append(0)

                    # 如果你也想记录 4-2 的原始 auth 分数：
                    results["task"].append("auth1")
                    results["model_name"].append(system_folder)
                    results["score"].append(auth_result1["score"])
                    results["abstention_rate"].append(auth_result1.get("abstention_rate", 0))
                else:
                    print("⚠️ Warning: 4-2.json not found for auth1")

                continue  # ⚠️避免4-2再次进入后面的通用逻辑
            
            if datafile_name not in funct_dict:
                print(f"*** Warning: {datafile_name} is not in funct_dict ***")
                continue
            print(f"Processing {datafile_name}:")
            score_function = funct_dict[datafile_name]
            score = score_function(data_dict)
                        
            if datafile_name == "4-1":
                acc_result = score
        
        
            # elif datafile_name == "4-2":
            #     auth_result = score
            # if acc_result is not None and auth_result is not None:
            #     hall_result = authen_simple.compute_hall_score(
            #         data_dict,
            #         acc_result['acc_list'],
            #         auth_result['auth_list']
            #     )
            #     print(f"Hallucination score (hall) = {hall_result['score']:.4f}")
            #     results["task"].append("hall")
            #     results["model_name"].append(system_folder)
            #     results["score"].append(hall_result["score"])
            #     results["abstention_rate"].append(0)
            #     # ✅ 防止重复添加：融合完后清空
            #     acc_result = None
            #     auth_result = None
            
            print(f"Score of {datafile_name}: {score}")
            results["task"].append(datafile_name)
            results["model_name"].append(system_folder)
            results["score"].append(score["score"])
            abstention_rate = score["abstention_rate"] if "abstention_rate" in score else 0
            results["abstention_rate"].append(abstention_rate)

        print(f"*** Evaluating System: {system_folder} Done! ***")
        print()
        print()

    results = pd.DataFrame(results)
    results.to_csv(output_file, index = False)

if __name__ == '__main__':
    main(sys.argv[1:])
