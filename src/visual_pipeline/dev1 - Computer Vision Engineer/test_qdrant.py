import torch
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient

model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)
device = "cpu"

client = QdrantClient(path="local_qdrant_db")

inputs = processor(text=["a group of people walking on the street"], return_tensors="pt", padding=True).to(device)

with torch.no_grad():
    outputs = model.get_text_features(**inputs)
    if isinstance(outputs, torch.Tensor):
        text_features = outputs
    elif hasattr(outputs, "text_embeds"):
        text_features = outputs.text_embeds
    else:
        text_features = outputs[0]

text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
query_vector = text_features.cpu().numpy()[0].tolist()

print("Type of query_vector:", type(query_vector))
print("Length of query_vector:", len(query_vector))
print("First element:", type(query_vector[0]), query_vector[0])

try:
    res = client.query_points(
        collection_name="kis_images",
        query=query_vector,
        limit=5
    )
    print("Success!", len(res.points))
except Exception as e:
    print("Error:", repr(e))
