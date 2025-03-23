import os
from flask import Flask, render_template, request, jsonify, send_file, url_for, redirect, session
from datetime import timedelta
import google.generativeai as genai
import re
from PIL import Image
from io import BytesIO
import base64
from loader import predict_image
from leffa_utils.utils import list_dir
import requests
from urllib.parse import quote  # Import the quote function for URL encoding
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from werkzeug.utils import secure_filename
import time
import http.client
import json

from dotenv import load_dotenv
load_dotenv()

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


# App config
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Setup Gemini API key (HARDCODED - NOT RECOMMENDED)
GOOGLE_API_KEY = "AIzaSyDGQRcVAHQO9XBdymjS45Kb-4do1tCMMiw"  # Replace with your actual API Key
genai.configure(api_key=GOOGLE_API_KEY)

# Create a Gemini model instance
model = genai.GenerativeModel('gemini-1.5-flash')

# Modified Prompt (for /recommend) - more general for flexibility
prompt_recommend = """Based on the person's body type in the image, recommend 3 or 4 clothing styles for both upper and bottom wear. Provide only the clothing styles. No colors, body type, or materials. Respond with simple plain words with no styles, symbols, asterisks, bullet points, numbers."""

example_dir = "./ckpts/examples"
person1_images = list_dir(f"{example_dir}/person1")
garment_images = list_dir(f"{example_dir}/garment")

# RapidAPI configuration
RAPIDAPI_KEY = "023cddea22mshb9ad4b9ca86ec5fp199288jsndc080b4c401b"
RAPIDAPI_HOST = "ebay38.p.rapidapi.com"

import gradio as gr

def convert_png_to_jpg(input_path, output_path):
    try:
        if isinstance(input_path, str):
            image = Image.open(input_path)
        elif isinstance(input_path, Image.Image):
            image = input_path
        
        if image.mode == 'RGBA':
            image = image.convert("RGB")
        
        image.save(output_path, format="JPEG")
    finally:
        print("Conversion completed.")

# Predefined clothing options
CLOTH_OPTIONS = {
    "Cloth 101": "/content/project/samplecloth/01_upper.jpg",
    "Cloth 102": "/content/project/samplecloth/tantra.jpg",
    "Cloth 103": "/content/project/samplecloth/09163_00.jpg",
    "Cloth 104": "/content/project/samplecloth/09176_00.jpg",
    "Cloth 105": "/content/project/samplecloth/09236_00.jpg",
    "Cloth 106": "/content/project/samplecloth/09290_00.jpg",
    "Cloth 107": "/content/project/samplecloth/clothing1.jpg",
    "Cloth 108": "/content/project/samplecloth/brown-jacket.jpg",
    "Cloth 109": "/content/project/samplecloth/boys-Puffer-Coat-With-Detachable-Hood-1.jpg",
    "Cloth 110": "/content/project/samplecloth/boys-Puffer-Coat-With-Detachable-Hood-2.jpg",
    "Cloth 111": "/content/g-polos-tshirt-2.png",
    "Cloth 112": "/content/project/samplecloth/tantra.jpg",
    "Cloth 113": "/content/project/samplecloth/pink-jacket.jpg",
    "Cloth 115": "/content/project/samplecloth/Man-Geox-Winter-jacket-1.jpg",
    "Cloth 116": "/content/project/samplecloth/black-hoodie.jpg",
    "Cloth 117": "/content/project/samplecloth/yellow-hoodie.png",
    "Cloth 118": "/content/project/samplecloth/purple-hoodie.jpg",
    "Cloth 119": "/content/project/samplecloth/b-shirt.jpg",
    "Cloth 120": "/content/project/samplecloth/blue-hoodie.jpg",
    "Cloth 121": "/content/project/samplecloth/hh1.png",
    "Cloth 122": "/content/project/samplecloth/wed-try.jpg",
    "Cloth 123": "/content/project/samplecloth/green-shirt.jpg",
    "Cloth 124": "/content/project/samplecloth/green-top.jpg",
    "Cloth 125": "/content/project/samplecloth/green-top2.jpg",
    "Cloth 126": "/content/project/samplecloth/black-top.jpg",
    "Cloth 127": "/content/project/samplecloth/black-top2.jpg",
    "Cloth 128": "/content/project/samplecloth/shorty.jpg"
}

def run(cloth_key, model):
    image_path = CLOTH_OPTIONS.get(cloth_key, None)
    if not image_path:
        return "Invalid Cloth Selection", None
    
    cloth = Image.open(image_path)
    model.save("temp_model.jpg")
    output_file = "temp_cloth2.jpg"
    convert_png_to_jpg(input_path=cloth, output_path=output_file)
    
    try:
        generated_image = predict_image(
            "temp_model.jpg", output_file, 
            vt_ref_acceleration=True, 
            vt_step=50, vt_scale=2.5, 
            vt_seed=42, vt_model_type="viton_hd", 
            vt_garment_type="upper_body", vt_repaint=False
        )
        return cloth, generated_image
    except Exception as e:
        return f"Error processing image: {str(e)}", None

with gr.Blocks() as demo:
    gr.Markdown("# DigiDrape Virtual Try-On System")

    cloth_key_input = gr.Dropdown(choices=list(CLOTH_OPTIONS.keys()), label="Select a Garment")
    human_image_input = gr.Image(label="Upload Human Image", type="pil")
    cloth_image_output = gr.Image(label="Selected Cloth")
    combined_image_output = gr.Image(label="Generated Try-On Image")

    submit_button = gr.Button("Combine and Display")
    submit_button.click(
        run,
        inputs=[cloth_key_input, human_image_input],
        outputs=[cloth_image_output, combined_image_output],
    )

demo.launch(share=True, debug=True)
