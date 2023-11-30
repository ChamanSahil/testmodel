# gradio
# openvino>=2023.1.0
# transformers[torch]>=4.30
import gradio
from openvino import inference_engine as ie
from openvino.inference_engine import IECore
from transformers import CLIPProcessor, CLIPModel, pipeline

# load pre-trained model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
# load preprocessor for model input
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

import requests
import numpy as np

from urllib.request import urlretrieve
from pathlib import Path
from PIL import Image

from flask import Flask
app = Flask(__name__)

DREAMS = ['I am going to win this event nad prove myself']
urlretrieve(
    "https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/228-clip-zero-shot-image-classification/visualize.py",
    filename='visualize.py'
)

# Function for processing the image
def processImage(imgURL, labels):
    sample_path = Path("data/coco.jpg")
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(
        imgURL,
        sample_path,
    )
    image = Image.open(sample_path)

    input_labels = labels
    text_descriptions = [f"This is a photo of a {label}" for label in input_labels]

    inputs = processor(text=text_descriptions, images=[image], return_tensors="pt", padding=True)

    results = model(**inputs)
    logits_per_image = results['logits_per_image']  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1).detach().numpy()  # we can take the softmax to get the label probabilities

    prob = logits_per_image.softmax(dim=1).detach().numpy()[0]
    result = np.array(prob)

    sorted_indices = np.argsort(result)
    sorted_result = result[sorted_indices]

    # Making a dictionary
    label_to_value = dict(zip(input_labels, prob))
    sorted_label_to_value = dict(sorted(label_to_value.items(), key=lambda item: item[1], reverse= True))
    first_3_results = dict(list(sorted_label_to_value.items())[:3])
    print(first_3_results)
    return first_3_results

url = 'https://versatilevats.com/openshift/labels.txt'
response = requests.get(url)

@app.route('/')
def build():
    if response.status_code == 200:
        labels = response.text.split('\n')
        print(labels)
        res = processImage(
            "https://versatilevats.com/squarehub/glitch/chamanrock145-Lawnmover.jpeg",
            labels
        )
        return res
    else:
        print("Error while fetching the labels")
        return "Error while fetching the labels"
  
@app.route('/dreams')
def dreams():
    return DREAMS
  
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
