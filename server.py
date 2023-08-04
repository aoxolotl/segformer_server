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

        # Pass image through pipeline
        outs = pipe(img)

        # Segformer spits out a list of masks with associated class name
        # We ignore the class preds for now
        # Use masks to build polygons
        # This polygon list will be passed on to the front end (PaperSegmentation)

        # Use only the final bin mask in the image
        mask_bin = outs[-1]["mask"]
        polygons = Mask(mask_bin).polygons().points
        polygons = [polygon.tolist() for polygon in polygons if len(polygon) > 2]
        end = time.time()
        process_time = end - start
        print(f"Process Time: {process_time:9.3} seconds", flush=True)

        return jsonify({"polygons": polygons})

    return "<h4>Model server is up<h4>"


if __name__ == "__main__":
    app.run()
