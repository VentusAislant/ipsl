from transformers import CLIPImageProcessor
from PIL import Image

img_processor = CLIPImageProcessor.from_pretrained('../pretrained_models/openai/clip-vit-large-patch14-336')
img = Image.open('cat.jpg').convert('RGB')
img_tensor = img_processor.preprocess(img, return_tensors='pt')['pixel_values'][0]
print(img_tensor.shape)
print(img_tensor.min(), img_tensor.max(), img_tensor.dtype)