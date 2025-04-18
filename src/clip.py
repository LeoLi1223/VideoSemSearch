import json
import threading
import torch
import cv2
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import numpy as np

last_saved_frame = [float("-inf")]  # 用列表包一层避免闭包问题
save_interval = 30  # 至少间隔 100 帧才允许保存
softmax_threshold = 0.995 # For Inclusion Query
exclude_threshold = 0.20 # For Exclusion Query
default_fps = 10.0
default_frame_skip = 5
# fps will be passed in dynamically

# Load CLIP model (ViT-B/32)
save_lock = threading.Lock()
model_version = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_version)
processor = CLIPProcessor.from_pretrained(model_version)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")
model.to(device)

# Create folder for matched images if not exists
output_folder = "matched_images"
max_images = 100
os.makedirs(output_folder, exist_ok=True)

# === Segment buffer for live aggregation ===
segment = {
    "start": None,  # start frame index
    "end": None,  # last included frame index
    "scores": [],  # list of softmax scores
}

# === Tracking ===
matched_frame_indices = []  # used for image saving
frame_softmax_hits = []  # for pred_relevant_windows: (frame, score)

# Clear the output folder before each run
for fname in os.listdir(output_folder):
    file_path = os.path.join(output_folder, fname)
    if os.path.isfile(file_path):
        os.remove(file_path)


def flush_segment_to_json(qid, raw_query, vid, fps, json_path="output_windows.json"):
    if segment["start"] is None:
        return

    start_time = round(segment["start"] / fps, 2)
    end_time = round((segment["end"] + 1) / fps, 2)
    avg_score = float(round(sum(segment["scores"]) / len(segment["scores"]), 4))

    data = {"qid": qid, "query": raw_query, "vid": vid, "pred_relevant_windows": []}

    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(
                f"Warning: '{json_path}' is corrupted or incomplete. Reinitializing..."
            )

    data["pred_relevant_windows"].append([start_time, end_time, avg_score])

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    # Reset segment
    segment["start"] = None
    segment["end"] = None
    segment["scores"] = []

def flush_query_header(qid, raw_query, vid, fps, json_path="output_windows.json"):
    data = {"qid": qid, "query": raw_query, "vid": vid, "pred_relevant_windows": []}

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


def run_clip(
    include_queries,
    exclude_queries,
    default_query,
    image,
    frame_index=0,
    frame_skip=default_frame_skip,
    fps=default_fps,
    json_path="output_windows.json",
    raw_query="user_query",
    qid="",
    vid="",
):
    global matched_frame_indices
    flush_query_header(qid, raw_query, vid, fps, json_path)

    # === Softmax
    def get_softmax_score_for_query(query):
        all_queries = [query] + default_query
        inputs = processor(
            text=all_queries, images=image, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        similarity = (outputs.image_embeds @ outputs.text_embeds.T).squeeze(0)
        similarity_scores = 100.0 * similarity.cpu().numpy()

        # softmax over [current predicate, default1, default2...]
        exp_scores = np.exp(similarity_scores - np.max(similarity_scores))
        softmax_scores = exp_scores / exp_scores.sum()

        return float(softmax_scores[0])  # Score for query

    include_scores = [(q, get_softmax_score_for_query(q)) for q in include_queries]
    exclude_scores = [(q, get_softmax_score_for_query(q)) for q in exclude_queries]

    # === [DEBUG]Print scores for frames
    # print(f"\n[DEBUG] Frame {frame_index}")
    # print("  Include scores:")
    # for q, s in include_scores:
    #     print(f"    {q}: {s:.4f}")
    # print("  Exclude scores:")
    # for q, s in exclude_scores:
    #     print(f"    {q}: {s:.4f}")

    include_pass = any(s > softmax_threshold for _, s in include_scores)
    exclude_fail = any(s > exclude_threshold for _, s in exclude_scores)

    # === Record all matches over threshold, even if not saved
    matched = include_pass and not exclude_fail
    if matched:
        score = max(s for _, s in include_scores)
        if segment["start"] is None:
            segment["start"] = frame_index
            segment["end"] = frame_index
            segment["scores"] = [float(score)]
        elif frame_index <= segment["end"] + frame_skip:
            segment["end"] = frame_index
            segment["scores"].append(float(score))
        else:
            flush_segment_to_json(qid, raw_query, vid, fps, json_path)
            segment["start"] = frame_index
            segment["end"] = frame_index
            segment["scores"] = [float(score)]
    else:
        flush_segment_to_json(qid, raw_query, vid, fps, json_path)

    # === Save image if matched
    if matched:
        with save_lock:
            if frame_index - last_saved_frame[0] < save_interval:
                return
            existing = sorted(
                [
                    fname
                    for fname in os.listdir(output_folder)
                    if fname.startswith("match_") and fname.endswith(".png")
                ]
            )
            if len(existing) < max_images:
                save_path = os.path.join(output_folder, f"match_{len(existing)+1}.png")
                image.save(save_path)
                last_saved_frame[0] = frame_index
                print(f"✅ Image saved to {save_path}")
            else:
                print(f"Maximum number of matched images ({max_images}) already saved.")


def finalize_clip_session(
    raw_query="user_query",
    qid="",
    vid="",
    fps=default_fps,
    json_path="output_windows.json",
):
    flush_segment_to_json(qid, raw_query, vid, fps, json_path)
    print(f"[INFO] Final segment flushed to {json_path}")
