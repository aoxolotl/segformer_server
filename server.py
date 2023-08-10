from transformers import pipeline
from flask import Flask, request, jsonify
from PIL import Image

import numpy as np
import time
import torch
import os

# Setup pipeline
# Use Segformer model
# Load model here
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
        points = content["points"]
        # Use full image if points is none or empty
        full_image = points is None or len(points) == 0

        img = Image.open(path)
        if not full_image:
            points = np.array(points)
            # Use points to define crop boundary
            l = points.min(axis=0)
            r = points.max(axis=0)
            img = img.crop((l[0], l[1], r[0], r[1]))

        # Pass image through pipeline
        outs = pipe(img)

        # Segformer spits out a list of masks with associated class name
        # We ignore the class preds for now
        # Use masks to build polygons
        # This polygon list will be passed on to the front end (PaperSegmentation)

        # Convert mask to numpy arrays
        masks = np.array([np.array(out["mask"]) for out in outs])

        # Do largest mask only if points exist
        if not full_image:
            largest_mask = np.argmax([np.sum(mask > 0) for mask in masks])
            masks = np.array([masks[largest_mask]])

        end = time.time()
        process_time = end - start
        print(f"{len(masks)} detected by Segformer", flush=True)
        print(f"Process Time: {process_time:9.3} seconds", flush=True)

        return jsonify({"masks": masks.tolist()})

    return "<h4>Model server is up<h4>"


if __name__ == "__main__":
    app.run()
