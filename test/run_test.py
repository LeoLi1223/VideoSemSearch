import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import argparse
from src_test.video_analytic_system import VideoAnalyticSystem
from src_test.prompt_processor import split_joined_predicates
from config.config import softmax_threshold, exclude_threshold
import numpy as np

print(softmax_threshold, exclude_threshold)

test_data_filepath = "../data/highlight_val_release.jsonl"
default_queries = ["mouse", "mug", "water bottle", "book", "orange", "computer", "gengar", "ghost", "phone", "bag",
                       "laptop","phone","backpack","keyboard","headphones","sofa", "television","bed","lamp","plant in a pot","person sitting at a table","window with sunlight","open fridge","bookshelf","mirror"]

single_output_json_path = "output_windows.json"
all_output_json_path = f"preds_{softmax_threshold}_{exclude_threshold}.jsonl"

latency_list = []


def run_one_test_split(test_data_json):
    qid = test_data_json["qid"]
    print(f"[INFO] qid: {qid}")
    user_query = test_data_json["query"]
    include, exclude = split_joined_predicates(user_query)
    print(f"[INFO] query: {user_query}")
    print(f"[INFO] include: {include}")
    print(f"[INFO] exclude: {exclude}")
    video_source = f"../data/{test_data_json['vid']}.mp4"
    print()

    system = VideoAnalyticSystem(
        video_source, include, exclude, default_queries, user_query, qid
    )
    latency = system.run()
    print(latency)
    latency_list.append(latency)

def run_one_test_no_split(test_data_json):
    qid = test_data_json["qid"]
    print(f"[INFO] qid: {qid}")
    user_query = test_data_json["query"]
    print(f"[INFO] query: {user_query}")
    video_source = f"../data/{test_data_json['vid']}.mp4"
    print()

    system = VideoAnalyticSystem(
        video_source, [user_query], [], default_queries, user_query, qid
    )
    system.run()


def run_val_data(num_test_data, args):
    with open(test_data_filepath, "r") as f:
        for _ in range(num_test_data):
            test_data = f.readline().strip("\n")
            test_data_json = json.loads(test_data)
            
            # choose split or not split
            if args.split:
                run_one_test_split(test_data_json)
            else:
                run_one_test_no_split(test_data_json)

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
    query_cnt = 0
    
    with open(ground_truth_filepath, 'r') as gt_file, open(preds_filepath, 'r') as pred_file:
        for gt_line, pred_line in zip(gt_file, pred_file):
            # Parse JSON from each line
            gt_data = json.loads(gt_line.strip())
            pred_data = json.loads(pred_line.strip())

            if gt_data["qid"] != pred_data["qid"]:
                print(f"[Error] Unmatch lines.")
                continue
            
            # Extract relevant information
            qid = pred_data["qid"]
            query = pred_data["query"]
            pred_windows = pred_data.get("pred_relevant_windows", [])
            gt_windows = gt_data.get("relevant_windows", [])

            if not gt_windows:
                print(f"Warning: No ground truth windows found for qid {qid}")
                continue
            
            if not gt_windows and not pred_windows:
                query_aps[query] = 1.0
                query_cnt += 1
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
                query_cnt += 1
            else:
                ap = 0.0
            
            # Store AP for this query
            query_aps[query] = ap
    
    # print(f"[INFO] {query_aps}")
    # print()

    # Calculate mAP
    if query_cnt > 0:
        mAP_exc = sum(query_aps.values()) / query_cnt
    else:
        mAP_exc = 0.0
    
    if len(query_aps) > 0:
        mAP_inc = sum(query_aps.values()) / len(query_aps)
    else:
        mAP_inc = 0.0
    
    return mAP_inc, mAP_exc

def compute_mAP_refined(ground_truth_filepath, preds_filepath):
    query_aps = {}
    query_f1 = {}
    query_P = {}
    query_R = {}
    query_cnt = 0
    
    with open(ground_truth_filepath, 'r') as gt_file, open(preds_filepath, 'r') as pred_file:
        for gt_line, pred_line in zip(gt_file, pred_file):
            # Parse JSON from each line
            gt_data = json.loads(gt_line.strip())
            pred_data = json.loads(pred_line.strip())

            if gt_data["qid"] != pred_data["qid"]:
                print(f"[Error] Unmatch lines.")
                continue
            
            # Extract relevant information
            qid = pred_data["qid"]
            query = pred_data["query"]
            pred_windows = pred_data.get("pred_relevant_windows", [])
            gt_windows = gt_data.get("relevant_windows", [])

            query_aps[query], query_f1[query], query_P[query], query_R[query] = calculate_average_precision(pred_windows, gt_windows)
    
    # Calculate mAP as the mean of all AP values
    mean_ap = sum(query_aps.values()) / len(query_aps) if query_aps else 0.0
    mean_f1 = sum(query_f1.values()) / len(query_f1) if query_f1 else 0.0
    P = sum(query_P.values()) / len(query_P) if query_P else 0.0
    R = sum(query_R.values()) / len(query_R) if query_R else 0.0

    return mean_ap, mean_f1, P, R

def calculate_average_precision(predictions, ground_truth):
    """
    Calculate Average Precision for temporal detection tasks with different granularity windows.
    
    Args:
        predictions: List of predicted windows, each in the form [start_time, end_time, score].
                    Will be sorted by score (highest score first).
                    These are typically smaller/more fine-grained windows.
        ground_truth: List of ground truth windows, each in the form [start_time, end_time, score].
                     These are typically larger/broader windows.
    
    Returns:
        Average Precision value.
    """
    if not predictions or not ground_truth:
        return 0.0, 0.0, 0.0, 0.0
    
    # Sort predictions by score in descending order
    predictions = sorted(predictions, key=lambda x: x[2], reverse=True)
    
    # Make a copy of ground truth to track how many predictions match each ground truth
    gt_matched_count = [0] * len(ground_truth)
    
    # Store precision and recall values at each prediction
    precisions = []
    recalls = []
    
    # Track true positives and false positives
    true_positives = 0
    false_positives = 0
    
    # For each prediction (in order of confidence/score)
    for pred in predictions:
        # Check if this prediction overlaps with any ground truth
        matched = False
        
        for gt_idx, gt in enumerate(ground_truth):
            if has_overlap(pred, gt):
                # This prediction overlaps with this ground truth
                true_positives += 1
                gt_matched_count[gt_idx] += 1
                matched = True
                break  # Only match to one ground truth to avoid double counting
        
        if not matched:
            # This is a false positive
            false_positives += 1
        
        # Calculate current precision and recall
        precision = true_positives / (true_positives + false_positives)
        
        # For recall, we count a ground truth as "detected" if at least one prediction overlaps with it
        detected_gt_count = sum(1 for count in gt_matched_count if count > 0)
        recall = detected_gt_count / len(ground_truth)
        
        precisions.append(precision)
        recalls.append(recall)
    
    # Calculate AP using the precision-recall curve
    # Using the all-point interpolation method
    ap = 0.0
    
    # Sort unique recall points
    unique_recalls = sorted(set(recalls))
    
    # For each recall threshold
    for recall_threshold in unique_recalls:
        # Find the maximum precision at recall levels >= recall_threshold
        precision_at_recall = max([p for p, r in zip(precisions, recalls) if r >= recall_threshold])
        
        # If this is the first point, use the full width
        if recall_threshold == unique_recalls[0]:
            width = recall_threshold
        else:
            # Otherwise, use the width between this recall and the previous one
            prev_recall = max([r for r in unique_recalls if r < recall_threshold])
            width = recall_threshold - prev_recall
        
        # Add the area of this rectangle to AP
        ap += width * precision_at_recall
    
    final_precision = true_positives / (true_positives + false_positives)
    final_recall = unique_recalls[-1]
    f1 = 2 * (final_precision * final_recall) / (final_precision + final_recall) if (final_precision + final_recall) > 0 else 0
    
    return ap, f1, final_precision, final_recall

def has_overlap(pred_window, gt_window):
    """
    Check if two temporal windows overlap at all.
    
    Args:
        window1: List or tuple of [start_time, end_time, score].
        window2: List or tuple of [start_time, end_time, score].
    
    Returns:
        Boolean indicating whether the windows overlap.
    """
    # Extract start and end times (ignore scores)
    start1, end1, _ = pred_window
    start2, end2 = gt_window
    
    # Check for overlap (if either window starts before the other ends)
    return start1 < end2 and start2 < end1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', action='store_true', help='split query')
    args = parser.parse_args()

    if os.path.exists(single_output_json_path):
        os.remove(single_output_json_path)
    if os.path.exists(all_output_json_path):
        os.remove(all_output_json_path)

    run_val_data(50, args)

    # Latency evaluation
    print(latency_list)
    # Compute mean latency
    mean_latency = np.mean(latency_list)

    # Compute P95 latency (95th percentile)
    p95_latency = np.percentile(latency_list, 95)
    print(f"Mean latency: {mean_latency:.2f} ms")
    print(f"P95 latency: {p95_latency:.2f} ms")


    # test_data_filepath = "combined_test.jsonl"
    # all_output_json_path = "results/preds_0.9_0.2_0.57_0.61_0.79_split.jsonl"
    mAP_inc, mAP_exc = compute_mAP(test_data_filepath, all_output_json_path)
    print(f"mAP (include 0): {mAP_inc}, mAP (exclude): {mAP_exc}")

    mAP_refined, f1, P, R = compute_mAP_refined(test_data_filepath, all_output_json_path)
    print(f"mAP refined: {mAP_refined}, f1 score: {f1}, precision: {P}, recall: {R}")

    # stylize the performance file
    os.rename(all_output_json_path, f"preds_{softmax_threshold}_{exclude_threshold}_{mAP_inc:.2f}_{mAP_exc:.2f}_{mAP_refined:.2f}_{'split' if args.split else 'no_split'}.jsonl")