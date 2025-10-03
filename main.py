from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import os
from datetime import datetime

app = FastAPI(
    title="Plant Disease Detector",
    description="🌱 AI-powered plant disease detection system",
    version="2.0.0"
)

# Allow all origins for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
try:
    model = load_model('retrained_model.h5')
    with open('retrained_classes.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    print(f"✅ Model loaded successfully with {len(class_names)} classes")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    # Try to download model if not found
    try:
        import gdown
        print("📥 Downloading model from Google Drive...")
        url = 'https://drive.google.com/file/d/1pNubAFqcZ8KTPThSBKcSHkrcCA9o51-D/view?usp=sharing'
        gdown.download(url, 'model.h5', quiet=False)
        model = load_model('model.h5')
        with open('classes.txt', 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        print("✅ Backup model loaded")
    except:
        raise RuntimeError("Failed to load any model")

# Disease solutions
disease_solutions = {
    "Pepper__bell___Bacterial_spot": "🦠 Use copper-based fungicides. Remove infected plants. Avoid overhead watering.",
    "Pepper__bell___healthy": "✅ Plant is healthy! Maintain good watering practices.",
    "Potato___Early_blight": "🍂 Apply fungicides containing chlorothalonil. Practice crop rotation.",
    "Potato___Late_blight": "🔥 Remove and destroy infected plants immediately. Apply metalaxyl-based fungicides.",
    "Potato___healthy": "✅ Plant is healthy! Maintain proper spacing.",
    "Tomato_Bacterial_spot": "🦠 Use copper sprays. Remove infected plants. Avoid wetting leaves.",
    "Tomato_Early_blight": "🍂 Apply chlorothalonil fungicides. Remove infected leaves.",
    "Tomato_Late_blight": "🔥 Destroy infected plants immediately. Apply systemic fungicides.",
    "Tomato_Leaf_Mold": "🍄 Improve air circulation. Reduce humidity. Apply copper-based fungicides.",
    "Tomato_Septoria_leaf_spot": "🔴 Remove affected leaves. Apply fungicides. Avoid overhead watering.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "🕷️ Use insecticidal soap or neem oil. Increase humidity.",
    "Tomato__Target_Spot": "🎯 Apply fungicides regularly. Practice crop rotation.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "🦠 Control whitefly population. Remove infected plants.",
    "Tomato__Tomato_mosaic_virus": "🦠 Remove and destroy infected plants. Disinfect tools.",
    "Tomato_healthy": "✅ Plant is healthy! Continue regular care.",
    "PlantVillage": "🌿 General plant image from dataset.",
    "unknown_leaf": "🌿 This plant species is not supported. Currently supporting: Tomato, Potato, Pepper plants only.",
    "not_plant": "🚫 This doesn't appear to be a plant leaf. Please upload a clear image of a plant leaf."
}

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for model prediction"""
    img = image.resize((128, 128))
    return np.array(img) / 255.0

# Simple HTML interface
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Plant Disease Detector</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .container { background: #f5f5f5; padding: 30px; border-radius: 10px; }
        h1 { color: #2e7d32; text-align: center; }
        .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; cursor: pointer; }
        .upload-area:hover { background: #e8f5e8; }
        button { background: #4caf50; color: white; padding: 15px 30px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        button:hover { background: #45a049; }
        button:disabled { background: #cccccc; cursor: not-allowed; }
        .result { margin-top: 20px; padding: 20px; border-radius: 5px; }
        .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .untrained { background: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
        .not-plant { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .error { background: #f5c6cb; color: #721c24; border: 1px solid #f1b0b7; }
        .loading { text-align: center; padding: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌱 Plant Disease Detector</h1>
        <p>Upload a plant leaf image to detect diseases</p>
        
        <div class="upload-area" onclick="document.getElementById('imageInput').click()">
            <h3>📁 Click to Upload Image</h3>
            <p>Supports JPG, PNG, JPEG formats</p>
        </div>
        
        <input type="file" id="imageInput" accept="image/*" style="display: none;">
        
        <button onclick="predict()" id="predictBtn" style="width: 100%;">Analyze Plant</button>
        
        <div class="loading" id="loading" style="display: none;">
            <p>🔍 Analyzing image...</p>
        </div>
        
        <div id="result"></div>
    </div>

    <script>
        async function predict() {
            const fileInput = document.getElementById('imageInput');
            const resultDiv = document.getElementById('result');
            const loadingDiv = document.getElementById('loading');
            const predictBtn = document.getElementById('predictBtn');
            
            if (!fileInput.files[0]) {
                alert('Please select an image first!');
                return;
            }
            
            // Show loading
            loadingDiv.style.display = 'block';
            resultDiv.style.display = 'none';
            predictBtn.disabled = true;
            
            try {
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                
                const response = await fetch('/predict/', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                // Display result
                resultDiv.innerHTML = `
                    <h3>${data.prediction}</h3>
                    <p><strong>${data.message}</strong></p>
                    ${data.solution ? `<p>${data.solution}</p>` : ''}
                    ${data.confidence ? `<p><em>Confidence: ${(data.confidence * 100).toFixed(1)}%</em></p>` : ''}
                `;
                
                // Set result style
                resultDiv.className = 'result ' + data.status;
                resultDiv.style.display = 'block';
                
            } catch (error) {
                resultDiv.innerHTML = `
                    <h3>Error</h3>
                    <p>Failed to analyze image. Please try again.</p>
                `;
                resultDiv.className = 'result error';
                resultDiv.style.display = 'block';
            } finally {
                loadingDiv.style.display = 'none';
                predictBtn.disabled = false;
            }
        }
        
        // Trigger file input when upload area is clicked
        document.getElementById('imageInput').addEventListener('change', function() {
            document.getElementById('predictBtn').disabled = false;
        });
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return html_content

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "classes_loaded": len(class_names),
        "timestamp": datetime.now().isoformat(),
        "environment": "production"
    }

@app.get("/classes")
async def get_classes():
    return {
        "classes": class_names,
        "count": len(class_names),
        "trained_plants": ["Tomato", "Potato", "Pepper"]
    }

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Predict plant disease from uploaded image
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            return {
                "status": "error",
                "message": "Please upload an image file (JPEG, PNG, etc.)",
                "prediction": "Invalid file type"
            }
        
        # Read and validate image
        image_data = await file.read()
        if len(image_data) == 0:
            return {
                "status": "error", 
                "message": "Empty file uploaded",
                "prediction": "Error"
            }
        
        try:
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception as e:
            return {
                "status": "error",
                "message": f"Invalid image file: {str(e)}",
                "prediction": "Error"
            }
        
        # Preprocess and predict
        img_array = preprocess_image(image)
        img_batch = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_batch, verbose=0)
        confidence = float(np.max(prediction))
        class_idx = np.argmax(prediction)
        class_name = class_names[class_idx]
        
        # Prepare response
        if class_name == "not_plant":
            return {
                "status": "not_plant",
                "message": "🚫 Not a plant leaf",
                "prediction": "Not a plant leaf",
                "solution": disease_solutions["not_plant"],
                "confidence": confidence
            }
        elif class_name == "unknown_leaf":
            return {
                "status": "untrained_leaf",
                "message": "🌿 Untrained plant species", 
                "prediction": "Untrained leaf species",
                "solution": disease_solutions["unknown_leaf"],
                "confidence": confidence
            }
        else:
            return {
                "status": "success",
                "message": "✅ Plant disease identified!",
                "prediction": class_name,
                "solution": disease_solutions.get(class_name, "Consult agricultural expert."),
                "confidence": confidence
            }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error processing image: {str(e)}",
            "prediction": "Error"
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🌱 Starting Plant Disease Detection API on port {port}...")
    print(f"📍 Production URL: https://plant-1-tnlq.onrender.com")
    uvicorn.run(app, host="0.0.0.0", port=port)