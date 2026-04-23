import os
import sys
import pandas as pd
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Import internal modules
from src.preprocess import DataPreprocess
from src.feature_engineering import FeatureEngineer
from src.exception import CustomException

app = FastAPI(title="Student Performance ML API", version="1.0.0")

# Enable CORS for Next.js communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
MODELS_PATH = "outputs/models"
PREPROCESSOR_FILE = os.path.join(MODELS_PATH, "preprocessors.pkl")

# Data structure for Single Prediction
class StudentData(BaseModel):
    school: str
    sex: str
    age: int
    address: str
    famsize: str
    Pstatus: str
    Medu: int
    Fedu: int
    Mjob: str
    Fjob: str
    reason: str
    guardian: str
    traveltime: int
    studytime: int
    failures: int
    schoolsup: str
    famsup: str
    paid: str
    activities: str
    nursery: str
    higher: str
    internet: str
    romantic: str
    famrel: int
    freetime: int
    goout: int
    Dalc: int
    Walc: int
    health: int
    absences: int
    G1: int
    G2: int

@app.get("/")
def home():
    return {"status": "Online", "message": "ML Audit API is ready for connection"}

@app.post("/predict")
def predict_single(data: StudentData):
    try:
        # 1. Load Preprocessors & Best Model
        if not os.path.exists(PREPROCESSOR_FILE):
            raise HTTPException(status_code=404, detail="Preprocessor not found. Please run main.py first.")
        
        pre_objs = joblib.load(PREPROCESSOR_FILE)
        # Assuming we use the top ranked model from our audit (e.g., Logistic Regression or Random Forest)
        # For simplicity, we'll try to load random_forest.pkl as default
        model_file = os.path.join(MODELS_PATH, "random_forest.pkl")
        if not os.path.exists(model_file):
             # Fallback to any available .pkl model
             models = [f for f in os.listdir(MODELS_PATH) if f.endswith('.pkl') and f != 'preprocessors.pkl']
             if not models:
                 raise HTTPException(status_code=404, detail="No trained models found.")
             model_file = os.path.join(MODELS_PATH, models[0])
        
        model = joblib.load(model_file)
        
        # 2. Convert input to DataFrame
        df = pd.DataFrame([data.dict()])
        
        # 3. Apply Phase 2 & 3 logic (Simplified for Inference)
        # In a real production environment, we should use a unified Pipeline object
        # But here we follow our modular architecture
        
        # Manual Encoding for Binary Columns (Matching Phase 2)
        # Note: In production, these encoders should also be saved/loaded
        # For this version, we'll apply a quick map or assume the preprocessor handles it
        
        # --- Simplified Inference Flow ---
        # Note: This part needs the exact same transformations as training
        # Since our FE module returns a ColumnTransformer, we use that.
        
        # Placeholder for complex inference logic - will be refined based on user needs
        return {
            "prediction": "Logic Ready",
            "model_used": os.path.basename(model_file),
            "note": "API initialized. Full inference mapping will be completed in the next sync."
        }

    except Exception as e:
        return {"error": str(e)}

@app.post("/audit-upload")
async def audit_file(file: UploadFile = File(...)):
    # Logic to receive CSV and run the full main.py pipeline
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    # Save temp file
    temp_path = f"data/raw/temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
    
    return {"message": f"File {file.filename} received and saved for audit."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
