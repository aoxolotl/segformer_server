from PIL import Image
from transformers import pipeline
#from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from imantics import Mask
import torch
import numpy as np

pipe = pipeline("image-segmentation", model="nvidia/segformer-b3-finetuned-ade-512-512")
image = Image.open('images/duckduck.jpg')
w, h = image.size
points = np.array([[533, 413], [1369, 853], [536, 879], [1364, 407]])
l = points.min(axis=0)
r = points.max(axis=0)
crop_image = image.crop((l[0], l[1], r[0], r[1]))
outs = pipe(crop_image)

masks = [np.array(out["mask"]) for out in outs]
largest_mask = np.argmax([np.sum(mask) for mask in masks])
#masks = [mask for mask in masks if np.sum(mask > 0) > 10000]
breakpoint()
polygons = Mask(masks[largest_mask]).polygons().points
polygons = [(polygon + l).tolist() for polygon in polygons if len(polygon) > 2]

print(len(polygons))
