from transformers import pipeline
from flask import Flask, request, jsonify
from PIL import Image

import numpy as np
import time
import torch
import os

from imantics import Mask

# Setup pipeline
# Use Segformer model
model_name = "nvidia/segformer-b3-finetuned-ade-512-512"
pipe = pipeline("image-segmentation", model=model_name)

# Setup endpoint
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def get_segmentation_mask():
    if request.method == "POST":
        start = time.time()
        content = request.json

        path = content["path"]
        points = np.array(content["points"])

        img = Image.open(path)
        # TODO: Use points to define crop boundary
        l = points.min(axis=0)
        r = points.max(axis=0)
        crop_img = img.crop((l[0], l[1], r[0], r[1]))

        # Pass image through pipeline
        outs = pipe(crop_img)

        # Segformer spits out a list of masks with associated class name
        # We ignore the class preds for now
        # Use masks to build polygons
        # This polygon list will be passed on to the front end (PaperSegmentation)

        # Use only the final bin mask in the image
        masks = [np.array(out["mask"]) for out in outs]
        largest_mask = np.argmax([np.sum(mask > 0) for mask in masks])
        polygons = Mask(masks[largest_mask]).polygons().points
        polygons = [(polygon + l).tolist() for polygon in polygons if len(polygon) > 2]

        end = time.time()
        process_time = end - start
        print(f"Process Time: {process_time:9.3} seconds", flush=True)

        return jsonify({"polygons": polygons})

    return "<h4>Model server is up<h4>"


if __name__ == "__main__":
    app.run()
