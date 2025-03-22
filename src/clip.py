import torch
import cv2
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np

# Load CLIP model (ViT-B/32)
model_version = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_version)
processor = CLIPProcessor.from_pretrained(model_version)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model.to(device)


def run_clip(text, image):
    # Preprocess inputs for CLIP
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True)

    # Move inputs to GPU if available
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)

        image_embeds = outputs.image_embeds  # (1, 512)
        text_embeds = outputs.text_embeds  # (N, 512)

    # Normalize embeddings
    image_embeds /= image_embeds.norm(dim=-1, keepdim=True)
    text_embeds /= text_embeds.norm(dim=-1, keepdim=True)

    # Compute cosine similarity
    similarity = (image_embeds @ text_embeds.T).squeeze()

    # Convert to numpy for readability
    similarity_scores = np.atleast_1d(similarity.cpu().numpy())

    print(text, similarity_scores)

    if similarity_scores[0] >= 0.3:
        image.save("matching.jpg")
