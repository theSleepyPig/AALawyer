# evaluate_fap.py
import json
import argparse

def calculate_metrics(prediction_file):
    with open(prediction_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_tp, total_fp, total_fn = 0, 0, 0

    for item in data:
        truth_set = set(item['ground_truth'])
        pred_set = set(item['predicted'])

        tp = len(truth_set.intersection(pred_set))
        fp = len(pred_set.difference(truth_set))
        fn = len(truth_set.difference(pred_set))

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision * 100,
        "recall": recall * 100,
        "f1_score": f1 * 100
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate FAP metrics from prediction file")
    parser.add_argument("prediction_file", type=str, help="Path to the prediction JSON file")
    args = parser.parse_args()

    metrics = calculate_metrics(args.prediction_file)
    print(f"Results for: {args.prediction_file}")
    print(f"  Precision: {metrics['precision']:.2f}%")
    print(f"  Recall:    {metrics['recall']:.2f}%")
    print(f"  F1-Score:  {metrics['f1_score']:.2f}%")