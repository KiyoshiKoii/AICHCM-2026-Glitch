from transformers import CLIPProcessor, CLIPModel

# Chạy 2 dòng này, thư viện sẽ tự động download model "openai/clip-vit-base-patch32" về máy (khoảng ~600MB)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("Loaded CLIP model successfully!")
