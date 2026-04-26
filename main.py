import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import os

from train import class_names, solutions

app = FastAPI()

#  Model load
model = load_model(r"C:\Users\annus\OneDrive\Documents\plant deases\plant_disease_model_v1.h5")

# Ensure temp folder exists
os.makedirs("temp", exist_ok=True)

# Image preprocessing
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    #img_array = img_array / 256.0
    return img_array

# ✅ Prediction API
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        file_location = f"temp/{file.filename}"

        # Save uploaded file
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Preprocess
        img_array = preprocess_image(file_location)

        # Prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Convert index → disease name
        disease_name = class_names[predicted_class]

        # Get solution
        solution = solutions.get(disease_name, "No solution available")

        # Delete temp file
        os.remove(file_location)

        return JSONResponse(content={
            "disease": disease_name,
            "solution": solution
        })

    except Exception as e:
        return JSONResponse(content={
            "error": str(e)
        })