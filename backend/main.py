from fastapi import FastAPI, UploadFile, File
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
import gdown

# Initialize FastAPI
app = FastAPI()

# Model and classes
MODEL_PATH = "model.h5"
CLASSES_PATH = "classes.txt"

# Google Drive model link (use your file ID)
FILE_ID = "1I0hGjq6B4UCA8JYvX1WTKR4V8jYcmKdM"
DRIVE_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)

# Load model and class names
model = load_model(MODEL_PATH)
with open(CLASSES_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Dictionary of disease → solution
disease_solutions = {
    "Pepper__bell___Bacterial_spot": "Use copper-based fungicides and avoid overhead irrigation.",
    "Pepper__bell___healthy": "Plant is healthy. Maintain good watering and soil nutrition.",
    "Potato___Early_blight": "Apply fungicides (chlorothalonil, mancozeb) and rotate crops.",
    "Potato___Late_blight": "Remove infected plants. Apply fungicides like metalaxyl or chlorothalonil.",
    "Potato___healthy": "Plant is healthy. Continue regular care.",
    "Tomato_Bacterial_spot": "Use copper sprays, avoid overhead irrigation, rotate crops.",
    "Tomato_Early_blight": "Apply fungicides (chlorothalonil, mancozeb), remove infected leaves.",
    "Tomato_Late_blight": "Use resistant varieties, apply fungicides (metalaxyl), destroy infected plants.",
    "Tomato_Leaf_Mold": "Improve air circulation, apply fungicides (chlorothalonil, copper).",
    "Tomato_Septoria_leaf_spot": "Remove affected leaves, apply fungicides (chlorothalonil, copper).",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Use insecticidal soap, neem oil, or miticides.",
    "Tomato__Target_Spot": "Apply fungicides and rotate crops.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Control whiteflies, remove infected plants, use resistant varieties.",
    "Tomato__Tomato_mosaic_virus": "Remove and destroy infected plants, disinfect tools.",
    "Tomato_healthy": "Plant is healthy. Maintain good care.",
}

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "🌱 Plant Disease Detection API is running!"}

# Prediction endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Load and preprocess image
    img = Image.open(file.file).convert("RGB")
    img = img.resize((128, 128))  # match training input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Model prediction
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    class_name = class_names[class_idx]

    # Get solution (if available)
    solution = disease_solutions.get(class_name, "No solution found for this disease.")

    return {
        "prediction": class_name,
        "solution": solution
    }
