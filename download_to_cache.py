from transformers import pipeline

# TODO use a config file to setup model_name parameter
model_name = "nvidia/segformer-b3-finetuned-ade-512-512"
pipe = pipeline("image-segmentation", model=model_name)
