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
save_interval = 30      # 至少间隔 100 帧才允许保存
softmax_threshold = 0.99
default_fps = 10.0
default_frame_skip = 5
# fps will be passed in dynamically

# Load CLIP model (ViT-B/32)
save_lock = threading.Lock()
model_version = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_version)
processor = CLIPProcessor.from_pretrained(model_version)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model.to(device)

# Create folder for matched images if not exists
output_folder = "matched_images"
max_images = 100
os.makedirs(output_folder, exist_ok=True)

# === Segment buffer for live aggregation ===
segment = {
    "start": None,  # start frame index
    "end": None,    # last included frame index
    "scores": []    # list of softmax scores
}

# === Tracking ===
matched_frame_indices = []  # used for image saving
frame_softmax_hits = []  # for pred_relevant_windows: (frame, score)

# Clear the output folder before each run
for fname in os.listdir(output_folder):
    file_path = os.path.join(output_folder, fname)
    if os.path.isfile(file_path):
        os.remove(file_path)
        
def flush_segment_to_json(query_name, fps, json_path="output_windows.json"):
    if segment["start"] is None:
        return

    start_time = round(segment["start"] / fps, 2)
    end_time = round((segment["end"] + 1) / fps, 2)
    avg_score = round(sum(segment["scores"]) / len(segment["scores"]), 4)

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
    else:
        data = {"query": query_name, "pred_relevant_windows": []}

    data["pred_relevant_windows"].append([start_time, end_time, avg_score])

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    # Reset segment
    segment["start"] = None
    segment["end"] = None
    segment["scores"] = []



def run_clip(text_list, image, frame_index=0, query_name="user_query", frame_skip = default_frame_skip, fps=default_fps, json_path="output_windows.json"):
    global matched_frame_indices

    # === Preprocess inputs for CLIP (batch text with one image)
    inputs = processor(text=text_list, images=image, return_tensors="pt", padding=True)

    # === Move inputs to GPU if available
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # === Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)

        image_embeds = outputs.image_embeds  # (1, 512)
        text_embeds = outputs.text_embeds    # (N, 512)

    # === Compute cosine similarity for each query → shape (N,)
    similarity = (image_embeds @ text_embeds.T).squeeze(0)

    # === Convert to CPU numpy for readability
    similarity_scores = 100.0 * similarity.cpu().numpy()

    # === Compose result dictionary
    result = {text: float(score) for text, score in zip(text_list, similarity_scores)}
    
    # === Softmax
    exp_scores = np.exp(similarity_scores - np.max(similarity_scores))
    softmax_scores = exp_scores / exp_scores.sum(axis=0)
    softmax = {text: float(score) for text, score in zip(text_list, softmax_scores)}

    # === Record all matches over threshold, even if not saved
    matched = False
    for score in softmax.values():
        if score > softmax_threshold:
            matched = True
            # extend current segment or flush and start new
            if segment["start"] is None:
                segment["start"] = frame_index
                segment["end"] = frame_index
                segment["scores"] = [score]
            # elif frame_index == segment["end"] + 1:
            elif frame_index <= segment["end"] + frame_skip:
                segment["end"] = frame_index
                segment["scores"].append(score)
            else:
                flush_segment_to_json(query_name, fps, json_path)
                segment["start"] = frame_index
                segment["end"] = frame_index
                segment["scores"] = [score]
            break # one match is enough to include frame

    if not matched:
        flush_segment_to_json(query_name, fps, json_path) 

    # === Save image if any score exceeds threshold
    if any(score > softmax_threshold for score in softmax.values()):
        # print(f"similarity scores: {result}")
        print(f"softmax scores: {softmax}")
        print(f"Softmax sum: {sum(softmax.values())}")
        print()
        with save_lock:

            if frame_index - last_saved_frame[0] < save_interval:
                print(f"[INFO] Skipping frame {frame_index} — too close to last saved frame {last_saved_frame[0]}")
                return  # 忽略这帧
            
            # print(f"score: {score}")
            existing = sorted([
                fname for fname in os.listdir(output_folder)
                if fname.startswith("match_") and fname.endswith(".png")
            ])
            if len(existing) < max_images:
                next_index = len(existing) + 1
                save_path = os.path.join(output_folder, f"match_{next_index}.png")
                image.save(save_path)

                last_saved_frame[0] = frame_index
                print(f"✅ Image saved to {save_path}")
            else:
                print("Maximum number of matched images (10) already saved.")

def finalize_clip_session(query_name="user_query", fps=default_fps, json_path="output_windows.json"):
    flush_segment_to_json(query_name, fps, json_path)
    print(f"[INFO] Final segment flushed to {json_path}")
