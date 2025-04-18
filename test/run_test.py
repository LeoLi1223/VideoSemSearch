import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import load_jsonl
import json
from src.video_analytic_system import VideoAnalyticSystem
from src.prompt_processor import split_joined_predicates

val_filepath = "data/highlight_val_release.jsonl"
default_queries = [
    "mouse",
    "mug",
    "water bottle",
    "book",
    "computer",
    "gengar",
    "ghost",
    "phone",
    "bag",
    "fruit basket",
    "table",
    "background",
    "buildings",
    "cars",
    "person",
    "woman",
    "man",
    "standing person",
    "sitting person",
    "face",
    "market vendor",
    "hand",
    "body",
    "crowd",
    "tree",
    "sky",
    "box",
    "sign",
    "poster",
    "camera",
]
single_output_json_path = "output_windows.json"
all_output_json_path = "val_preds.jsonl"


def run_one_test(test_data_json):
    qid = test_data_json["qid"]
    print(f"[INFO] qid: {qid}")
    user_query = test_data_json["query"]
    include, exclude = split_joined_predicates(user_query)
    print(f"[INFO] query: {user_query}")
    print(f"[INFO] include: {include}")
    print(f"[INFO] exclude: {exclude}")
    video_source = f"data/{test_data_json['vid']}.mp4"
    print(f"[INFO] vid: {test_data_json['vid']}")

    system = VideoAnalyticSystem(
        video_source, include, exclude, default_queries, user_query, qid
    )
    system.run()


def run_val_data(num_test_data):
    with open(val_filepath, "r") as f:
        for _ in range(num_test_data):
            test_data = f.readline().strip("\n")
            test_data_json = json.loads(test_data)
            run_one_test(test_data_json)

            if not os.path.exists(all_output_json_path):
                open(all_output_json_path, "x").close()

            if os.path.exists(single_output_json_path):
                with open(single_output_json_path, "r") as single_output_json_file:
                    data = json.load(single_output_json_file)
                with open(all_output_json_path, "a") as all_output_json_file:
                    all_output_json_file.write(json.dumps(data) + "\n")
                os.remove(single_output_json_path)


def compute_mAP(ground_truth_filepath, preds_filepath):
    # Dictionary to store AP for each query
    query_aps = {}
    
    with open(ground_truth_filepath, 'r') as gt_file, open(preds_filepath, 'r') as pred_file:
        for gt_line, pred_line in zip(gt_file, pred_file):
            # Parse JSON from each line
            gt_data = json.loads(gt_line.strip())
            pred_data = json.loads(pred_line.strip())
            
            # Extract relevant information
            qid = pred_data["qid"]
            query = pred_data["query"]
            pred_windows = pred_data.get("pred_relevant_windows", [])
            gt_windows = gt_data.get("relevant_windows", [])
            
            if not gt_windows and not pred_windows:
                query_aps[query] = 1.0
                # print(f"Warning: No ground truth windows found for qid {qid}")
                continue
                
            # Sort predictions by confidence score (descending)
            # pred_windows.sort(key=lambda x: x[2], reverse=True)
            
            # Calculate precision at each position
            num_relevant = 0
            precisions = []
            
            for i, pred_window in enumerate(pred_windows):
                start_time, end_time, _ = pred_window
                
                # Check if prediction overlaps with any ground truth window
                is_relevant = False
                for gt_start, gt_end in gt_windows:
                    # Check for overlap
                    if max(start_time, gt_start) <= min(end_time, gt_end):
                        is_relevant = True
                        break
                
                if is_relevant:
                    num_relevant += 1
                    precision = num_relevant / (i + 1)
                    precisions.append(precision)
            
            # Calculate AP
            if num_relevant > 0:
                ap = sum(precisions) / num_relevant
            else:
                ap = 0.0
            
            # Store AP for this query
            query_aps[query] = ap
    
    print(query_aps)

    # Calculate mAP
    if len(query_aps) > 0:
        mAP = sum(query_aps.values()) / len(query_aps)
    else:
        mAP = 0.0
    
    return mAP


if __name__ == "__main__":
    if os.path.exists(single_output_json_path):
        os.remove(single_output_json_path)

    run_val_data(2)
    mAP = compute_mAP("data/highlight_val_release.jsonl", "val_preds.jsonl")
    print(f"mAP: {mAP}")