from PIL import Image
from transformers import pipeline
#from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from imantics import Mask
import torch

pipe = pipeline("image-segmentation", model="nvidia/segformer-b3-finetuned-ade-512-512")
image = Image.open('images/duckduck.jpg')
w, h = image.size
crop_image = image.crop((500, 500, w - 500, h - 500))
outs = pipe(crop_image)

polygons = []
for out in outs:
    polygons.append(Mask(out['mask']).polygons().points)

print(len(polygons))

# TODO: Use points as a crop roi measure

#device = "cpu"
#model_name = "nvidia/segformer-b3-finetuned-ade-512-512"
#feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
#model = SegformerForSemanticSegmentation.from_pretrained(model_name)
#model.to(device)
#
## Load image
#impath = 'images/duckduck.jpg'
#image = Image.open(impath)
#
## Process image
#pix_vals = feature_extractor(image, return_tensors="pt").pixel_values.to(device)
#
## Forward Pass: Get output masks
#outputs = model(pix_vals)
#logits = outputs.logits

# Visualize

# Convert them to Segmentations from datatorch api
## 1. Convert masks to polygons using imantics mask
##  a. Refer to server.py in dextr_server
## 2. Pass these polygons to function
