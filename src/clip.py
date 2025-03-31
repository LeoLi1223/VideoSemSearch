import threading
import torch
import cv2
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import numpy as np

last_saved_frame = [float("-inf")]  # 用列表包一层避免闭包问题
save_interval = 100      # 至少间隔 100 帧才允许保存

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
max_images = 10
os.makedirs(output_folder, exist_ok=True)

# Clear the output folder before each run
for fname in os.listdir(output_folder):
    file_path = os.path.join(output_folder, fname)
    if os.path.isfile(file_path):
        os.remove(file_path)

def run_clip(text_list, image, frame_index=0):
    # Preprocess inputs for CLIP (batch text with one image)
    inputs = processor(text=text_list, images=image, return_tensors="pt", padding=True)

    # Move inputs to GPU if available
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)

        image_embeds = outputs.image_embeds  # (1, 512)
        text_embeds = outputs.text_embeds    # (N, 512)

    # Compute cosine similarity for each query → shape (N,)
    similarity = (image_embeds @ text_embeds.T).squeeze(0)

    # Convert to CPU numpy for readability
    similarity_scores = 100.0 * similarity.cpu().numpy()

    # Compose result dictionary
    result = {text: float(score) for text, score in zip(text_list, similarity_scores)}
    
    # Softmax
    exp_scores = np.exp(similarity_scores - np.max(similarity_scores))
    softmax_scores = exp_scores / exp_scores.sum(axis=0)
    softmax = {text: float(score) for text, score in zip(text_list, softmax_scores)}

    print(f"similarity scores: {result}")
    print(f"softmax scores: {softmax}")
    print()

    # save image if any score exceeds threshold
    # if any(score > 28 for score in result.values()):
    if any(score > 0.90 for score in softmax.values()):
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
