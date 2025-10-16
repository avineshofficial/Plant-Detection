# app.py
import os
import json
import numpy as np
from flask import Flask, request, render_template, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Load the Trained Model and Class Names ---
MODEL_PATH = 'plant_disease_model.h5'
model = load_model(MODEL_PATH)

# Load class names from the json file
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

# --- Preprocess Function ---
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Define Routes ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    if file:
        filepath = os.path.join('static/uploads', file.filename)
        file.save(filepath)

        processed_img = preprocess_image(filepath)
        prediction = model.predict(processed_img)
        
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class_index]
        
        image_url = url_for('static', filename='uploads/' + file.filename)

        return render_template('result.html', prediction=predicted_class_name, image_url=image_url)

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)