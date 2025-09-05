from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel
import model_utils as mu
import json
import re
from typing import Dict, Any, Optional
import requests

# Add Keras 3 specific imports for loading
from keras.models import load_model # Import from keras directly
# These might not be strictly necessary if load_model handles them,
# but good for explicit environment setup
from keras.losses import MeanSquaredError as KerasLossMSE
from keras.metrics import MeanSquaredError as KerasMetricMSE
from keras.saving import register_keras_serializable
from tcn import TCN # Crucial for TCN model loading, ensure it's imported here too


import tensorflow as tf
print(f"api.py - TensorFlow Version: {tf.__version__}")
print(f"api.py - Keras Version: {tf.keras.__version__}")

# --- 1. Create a SINGLE FastAPI application instance ---
app = FastAPI(title="HGL Prediction API")

# --- 2. Add CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. Serve static files (HTML, CSS, JS) ---
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- 4. Serve the main HTML file at root ---
@app.get("/")
def get_root():
    return FileResponse("static/index.html")

# --- 5. API endpoint for testing ---
@app.get("/api/health")
def get_health():
    return {"message": "The API is running successfully!"}

# --- 5.1 Serve the template CSV for download ---
@app.get("/template.csv")
def get_template():
    return FileResponse(
        "static/template.csv",            # path to your file
        media_type="text/csv",
        filename="template.csv"           # forces browser download with this name
    )

# ===================================================================
# AI ASSISTANT IMPLEMENTATION
# ===================================================================

class ChatMessage(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None

from enhanced_assistant import enhanced_assistant

# ===================================================================
# EXISTING ENDPOINTS (UNCHANGED)
# ===================================================================

# --- 6. Define the endpoint for training a single model ---
@app.post("/train_single_model/")
async def train_single_model(
    model_name: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Accepts a CSV file and a model name, runs the pipeline for that single model,
    and returns the results.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV.")

    try:
        print(f"Received request to train model: {model_name}")
        
        # This function from model_utils.py returns four values
        metrics, predictions, geographic_data, error = mu.run_single_model_pipeline(file.file, model_name)

        if error:
            raise HTTPException(status_code=400, detail=error)
        
        print(f"Training for {model_name} complete. Sending results.")
        # Envoi des résultats au webhook n8n
        n8n_webhook_url = "https://khzouhai.app.n8n.cloud/webhook-test/hgl"  # Remplace par l'URL réelle de ton webhook n8n
        payload = {
            "model_name": model_name,
            "metrics": metrics,
            "predictions": predictions,
            "geographic_data": geographic_data
        }
        try:
            n8n_response = requests.post(n8n_webhook_url, json=payload, timeout=10)
            n8n_status = n8n_response.status_code
        except Exception as e:
            n8n_status = f"Erreur lors de l'envoi à n8n: {e}"

        return {
            "message": f"Training for {model_name} completed successfully!",
            "model_name": model_name,
            "metrics": metrics,
            "predictions": predictions,
            "geographic_data": geographic_data,
            "n8n_status": n8n_status
        }

    except Exception as e:
        print(f"An unexpected error occurred in /train_single_model/: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected internal server error occurred: {str(e)}")


# --- 7. Define the endpoint for comparing all models ---
@app.post("/train_all_models/")
async def train_all_models(file: UploadFile = File(...)):
    """
    Accepts a CSV file, runs the pipeline for ALL models, and returns
    the metrics AND predictions for a full comparison view.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV.")

    try:
        print("Received request to train ALL models for comparison.")
        
        # This function from model_utils.py returns four values
        all_metrics, all_predictions, geographic_data, error = mu.run_all_models_pipeline(file.file)

        if error:
            raise HTTPException(status_code=400, detail=error)
        
        print("All models trained. Sending comparison metrics and predictions.")
        # Envoi des résultats au webhook n8n
        n8n_webhook_url = "https://khzouhai.app.n8n.cloud/webhook-test/hgl"  # Remplace par l'URL réelle de ton webhook n8n
        payload = {
            "metrics": all_metrics,
            "predictions": all_predictions,
            "geographic_data": geographic_data
        }
        try:
            n8n_response = requests.post(n8n_webhook_url, json=payload, timeout=10)
            n8n_status = n8n_response.status_code
        except Exception as e:
            n8n_status = f"Erreur lors de l'envoi à n8n: {e}"

        return {
            "message": "All models trained successfully!",
            "metrics": all_metrics,
            "predictions": all_predictions,
            "geographic_data": geographic_data,
            "n8n_status": n8n_status
        }

    except Exception as e:
        print(f"An unexpected error occurred in /train_all_models/: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected internal server error occurred: {str(e)}")

@app.post("/chat")
async def chat_with_assistant(chat_message: ChatMessage):
    """
    Enhanced chat endpoint with intelligent analysis capabilities
    """
    try:
        response = enhanced_assistant.generate_response(
            chat_message.message, 
            chat_message.context
        )
        
        return {
            "response": response,
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Assistant error: {str(e)}"
        )
