import os
import logging
import gdown
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
import numpy as np
import tensorflow as tf
import cv2
tf.config.set_visible_devices([], 'GPU')

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Path dan Google Drive File ID
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "model.keras")
GOOGLE_DRIVE_FILE_ID = "1cLQCO3DfLDYMwyYqrwylwOmszJM2PUjJ"  # <-- Ganti dengan file ID kamu

# Fungsi untuk download model jika belum ada
def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        logger.info("📥 Model not found locally. Downloading from Google Drive...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        logger.info("✅ Model downloaded successfully.")

# Download jika model belum ada
download_model_if_needed()

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error("Error loading model: %s", e)
    raise

# Inisialisasi FastAPI
app = FastAPI()

# Allow CORS
origins = [
    "http://localhost:5173",  # React Vite default
    "http://localhost",
    "http://localhost:8080",
    "https://dental-caries-classification.netlify.app",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fungsi preprocessing
def preprocess_image(image: Image.Image) -> np.ndarray:
    # Ubah ke ukuran 224x224
    img = image.resize((224, 224))
    
    # Konversi ke array dan ubah ke format BGR (untuk OpenCV)
    img_array = np.array(img)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Terapkan bilateral filtering (lebih cocok untuk X-ray dental)
    denoised = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Kembali ke RGB (jika model dilatih dengan RGB)
    denoised_rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
    
    # Normalisasi dan tambahkan dimensi batch
    normalized = denoised_rgb / 255.0
    img_ready = np.expand_dims(normalized, axis=0)
    
    return img_ready

# Endpoint utama
@app.post("/api/v1/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        preprocessed_image = preprocess_image(image)
    except UnidentifiedImageError as e:
        logger.error("Uploaded file is not a valid image: %s", e)
        raise HTTPException(status_code=400, detail="Invalid image file")
    except Exception as e:
        logger.error("Error processing image file: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")

    try:
        prediction = model.predict(preprocessed_image)
        predicted_class = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))

        class_names = [
            "Class I Caries",
            "Class II Caries",
            "Class III Caries",
            "Class IV Caries",
            "Class V Caries",
            "Class VI Caries",
            "Non-Caries",
        ]

        descriptions = [
            "Caries located in pits and fissures on the occlusal surfaces of molars and premolars, or in the lingual pits of maxillary incisors.",
            "Caries found on the proximal (mesial or distal) surfaces of posterior teeth, including molars and premolars.",
            "Caries on the proximal surfaces of anterior teeth without involvement of the incisal angle.",
            "Caries on the proximal surfaces of anterior teeth involving the incisal angle.",
            "Caries located in the cervical third of the facial or lingual surfaces of both anterior and posterior teeth.",
            "Caries involving the incisal edges of anterior teeth or the cusp tips of posterior teeth.",
            "No visible carious lesion detected. This may include healthy teeth or non-carious conditions such as developmental anomalies, fluorosis, or minor enamel defects.",
        ]

        recommendations = [
            "Prompt restoration is recommended using composite resin or amalgam, depending on the depth and accessibility of the lesion.",
            "Requires radiographic evaluation and restoration using composite resin or amalgam. Consider preventive measures for adjacent contacts.",
            "Aesthetic restoration using composite resin is recommended to preserve function and appearance.",
            "Immediate restoration with composite resin buildup is necessary. A crown may be indicated for extensive damage.",
            "Restoration with glass ionomer cement (GIC) or composite resin is advised. Periodic monitoring of gingival health is essential.",
            "Restoration with composite resin is recommended. Occlusal analysis may be needed to identify contributing parafunctional habits.",
            "Maintain proper oral hygiene and continue routine dental check-ups every 6 months.",
        ]

        result = {
            "caries_class": class_names[predicted_class],
            "description": descriptions[predicted_class],
            "recommendation": recommendations[predicted_class],
            "confidence_level": confidence,
        }

        logger.info("Prediction result: %s", result)
        return JSONResponse(content=result)

    except Exception as e:
        logger.error("Error during prediction: %s", e)
        raise HTTPException(status_code=500, detail="Error processing image")

# Untuk run manual jika dijalankan langsung
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")

