from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np

model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
inputs = processor(images=image, return_tensors="pt")

out1 = model.get_image_features(**inputs)
print("Type of out1:", type(out1))
if hasattr(out1, "shape"):
    print("Shape:", out1.shape)
else:
    print("Dir:", dir(out1))

out2 = model(**inputs)
print("Type of out2:", type(out2))
if hasattr(out2, "image_embeds"):
    print("image_embeds shape:", out2.image_embeds.shape)
