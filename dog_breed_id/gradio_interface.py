# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/07_gradio_interface.ipynb.

# %% auto 0
__all__ = ['detector', 'app', 'process']

# %% ../nbs/07_gradio_interface.ipynb 2
from .inference import *
import gradio as gr
from PIL import Image

detector = DogBreedDetector('resnet50.pt', 'model-fasterrcnn.cuda.pt', 'id2labels.json', 'label2id.json')

def process(img):
    img = Image.fromarray(img)
    preds = detector(img)
    annotation = annotate_prediction(img, preds)
    return annotation

app = gr.Interface(fn=process, inputs=['image'], outputs=['image'], description='Take or upload the image of a dog to detect breed')
app.launch()