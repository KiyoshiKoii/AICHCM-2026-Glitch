import os
import sys
import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)
device = "cpu"
model.to(device)

script_dir = os.path.dirname(os.path.abspath(__file__))

# Lay npy cua video dau tien
npy_dir = os.path.join(script_dir, "npy_features")
npy_files = sorted([f for f in os.listdir(npy_dir) if f.endswith(".npy")])
first_npy = npy_files[0]
video_name = first_npy.replace(".npy", "")
saved_vectors = np.load(os.path.join(npy_dir, first_npy))
print(f"Video: {video_name}, saved vectors shape: {saved_vectors.shape}")
print(f"Saved vector[0] norm: {np.linalg.norm(saved_vectors[0]):.6f}")
print(f"Saved vector[0][:10]: {saved_vectors[0][:10]}")

# Lay frame dau tien cua video do
keyframe_dir = os.path.join(script_dir, "keyframe", video_name)
frames = sorted([f for f in os.listdir(keyframe_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
first_frame_path = os.path.join(keyframe_dir, frames[0])

img = Image.open(first_frame_path).convert("RGB")
inputs = processor(images=[img], return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.get_image_features(**inputs)
    
    print(f"\nType of get_image_features output: {type(outputs)}")
    
    if isinstance(outputs, torch.Tensor):
        print("-> Tensor output! Shape:", outputs.shape)
        image_features = outputs
    elif hasattr(outputs, "pooler_output"):
        pool = outputs.pooler_output
        print(f"-> pooler_output shape: {pool.shape}")
        print(f"visual_projection in_features: {model.visual_projection.in_features}")
        if pool.shape[-1] == model.visual_projection.in_features:
            image_features = model.visual_projection(pool)
            print("-> Used visual_projection(pooler_output)")
        else:
            image_features = pool
    
    # L2 normalize
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    computed_vec = image_features.cpu().numpy()[0]
    
print(f"\nComputed vector norm: {np.linalg.norm(computed_vec):.6f}")
print(f"Computed vector[:10]: {computed_vec[:10]}")
print(f"\nCosine sim (computed vs saved[0]): {np.dot(computed_vec, saved_vectors[0]):.6f}")
print(f"\n=> Nếu cos sim ~1.0 thi pipeline dung. Nếu thap thi pipeline sai!")

# Bay gio test text encoding voi cung pipeline
print("\n--- Test TEXT encoding ---")
text_prompt = "TV news"
txt_inputs = processor(text=[text_prompt], return_tensors="pt", padding=True).to(device)

with torch.no_grad():
    txt_outputs = model.get_text_features(**txt_inputs)
    print(f"Type of get_text_features output: {type(txt_outputs)}")
    
    if isinstance(txt_outputs, torch.Tensor):
        text_features = txt_outputs
        print("-> Tensor output!")
    elif hasattr(txt_outputs, "pooler_output"):
        txt_pool = txt_outputs.pooler_output
        print(f"-> pooler_output shape: {txt_pool.shape}")
        print(f"text_projection in_features: {model.text_projection.in_features}")
        if txt_pool.shape[-1] == model.text_projection.in_features:
            text_features = model.text_projection(txt_pool)
            print("-> Used text_projection(pooler_output)")
        else:
            text_features = txt_pool
    
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    text_vec = text_features.cpu().numpy()[0]

# Compute cosine sim voi tat ca vectors trong DB cua video nay
sims = saved_vectors @ text_vec
top5_idx = np.argsort(sims)[-5:][::-1]
print(f"\nTop-5 most similar frames for '{text_prompt}':")
for idx in top5_idx:
    print(f"  {frames[idx]} (index {idx}), score={sims[idx]:.4f}")
