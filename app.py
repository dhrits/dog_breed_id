from dog_breed_id.inference import *
import gradio as gr
from PIL import Image
import os
import neptune

NEPTUNE_KEY = os.getenv('NEPTUNE_KEY')
run = neptune.init_run(
    project="dhrits/dog-breed-id",
    api_token=NEPTUNE_KEY,
)
detector = DogBreedDetector('resnet50.pt', 'model-fasterrcnn.cuda.pt', 'id2labels.json', 'label2id.json')

def process(img):
    img = Image.fromarray(img)
    preds = detector(img)
    label, confidence, box = preds
    run['confidence'].append(confidence)
    label = label.replace('_', ' ').capitalize()
    annotation = annotate_prediction(img, preds)
    return annotation, label, confidence

app = gr.Interface(
    fn=process, inputs=['image'], 
    outputs=[gr.Image(), gr.Label(label='Breed', value='N/A'), gr.Number(value=0, label='Confidence')], 
    description='Take or upload the image of a dog to detect breed'
)

if __name__ == '__main__':
    app.launch(share=True)

run.stop()
