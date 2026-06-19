import os
import sys
import pandas as pd
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io

app = FastAPI(title="Student Performance ML API", version="1.0.0")

# Enable CORS for Next.js communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_PATH = "outputs/models"
PREPROCESSOR_FILE = os.path.join(MODELS_PATH, "preprocessors.pkl")

# Pydantic schema matching the 31 features from the kaggle_higher_ed dataset
class StudentData(BaseModel):
    age: int
    sex: int
    graduated_h_school_type: int
    scholarship_type: int
    additional_work: int
    activity: int
    partner: int
    total_salary: int
    transport: int
    accomodation: int
    mother_ed: int
    farther_ed: int
    siblings: int
    parental_status: int
    mother_occup: int
    father_occup: int
    weekly_study_hours: int
    reading_non_scientific: int
    reading_scientific: int
    attendance_seminars_dep: int
    impact_of_projects: int
    attendances_classes: int
    preparation_midterm_company: int
    preparation_midterm_time: int
    taking_notes: int
    listenning: int
    discussion_improves_interest: int
    flip_classrom: int
    grade_previous: int
    grade_expected: int
    course_id: int

@app.get("/")
def home():
    return {
        "status": "Online",
        "message": "Student Performance ML API is active.",
        "preprocessor_exists": os.path.exists(PREPROCESSOR_FILE)
    }

@app.get("/models")
def get_models():
    """List all available models in outputs/models."""
    if not os.path.exists(MODELS_PATH):
        return {"models": []}
        
    all_files = [f for f in os.listdir(MODELS_PATH) if f.endswith('.pkl') and f != 'preprocessors.pkl']
    
    # Optional fallback if directory somehow has no models but files exist (should not happen)
    if not all_files:
        for f in ["random_forest.pkl", "gradient_boosting.pkl", "xgboost.pkl"]:
            if os.path.exists(os.path.join(MODELS_PATH, f)):
                all_files.append(f)
                
    return {"models": sorted(list(set(all_files)))}

@app.post("/predict")
def predict_single(data: StudentData, model_name: str = "random_forest.pkl"):
    try:
        # 1. Load Preprocessors
        if not os.path.exists(PREPROCESSOR_FILE):
            raise HTTPException(status_code=404, detail="Preprocessor file not found.")
        preprocessors = joblib.load(PREPROCESSOR_FILE)
        
        # 2. Load Model
        model_file = os.path.join(MODELS_PATH, model_name)
        if not os.path.exists(model_file):
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found.")
        model = joblib.load(model_file)
        
        # 3. Convert input to DataFrame
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])
        
        # 4. Transform using preprocessing pipeline
        ct = preprocessors['column_transformer']
        scaler = preprocessors['scaler']
        selector = preprocessors['selector']
        
        X_ct = ct.transform(df)
        X_scaled = scaler.transform(X_ct)
        X_selected = selector.transform(X_scaled)
        
        # 5. Predict
        prediction = model.predict(X_selected)[0]
        
        # 6. Probabilities (if available)
        probabilities = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_selected)[0]
            probabilities = {f"Grade {i}": float(p) for i, p in enumerate(probs)}
            
        return {
            "prediction": int(prediction),
            "probabilities": probabilities,
            "model_used": model_name
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/audit-upload")
async def audit_file(file: UploadFile = File(...), model_name: str = "random_forest.pkl"):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    try:
        # Load Preprocessors and Model
        if not os.path.exists(PREPROCESSOR_FILE):
             raise HTTPException(status_code=404, detail="Preprocessor file not found.")
        preprocessors = joblib.load(PREPROCESSOR_FILE)
        
        model_file = os.path.join(MODELS_PATH, model_name)
        if not os.path.exists(model_file):
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found.")
        model = joblib.load(model_file)
        
        # Read CSV file (handle potential semicolon or comma separator)
        contents = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(contents), sep=None, engine='python')
        except Exception:
            df = pd.read_csv(io.BytesIO(contents))
            
        # Ensure student_id is kept for final presentation but dropped for prediction
        student_ids = None
        if 'student_id' in df.columns:
            student_ids = df['student_id']
            df_predict = df.drop('student_id', axis=1)
        elif 'STUDENT_ID' in df.columns:
            student_ids = df['STUDENT_ID']
            df_predict = df.drop('STUDENT_ID', axis=1)
        else:
            df_predict = df.copy()
            
        # Target column drop if exists
        target_candidates = ['grade', 'Grade', 'G3', 'Target', 'status']
        for col in target_candidates:
            if col in df_predict.columns:
                df_predict = df_predict.drop(col, axis=1)
                
        # Transform
        ct = preprocessors['column_transformer']
        scaler = preprocessors['scaler']
        selector = preprocessors['selector']
        
        X_ct = ct.transform(df_predict)
        X_scaled = scaler.transform(X_ct)
        X_selected = selector.transform(X_scaled)
        
        # Predict
        preds = model.predict(X_selected)
        
        # Add predictions to original dataframe
        df['Predicted_Grade'] = [int(p) for p in preds]
        
        # Calculate summary statistics
        value_counts = df['Predicted_Grade'].value_counts().to_dict()
        summary = {f"Grade {k}": int(v) for k, v in value_counts.items()}
        
        # Convert df back to CSV for download
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return {
            "message": "Audit completed successfully",
            "summary": summary,
            "total_records": len(df),
            "csv_content": output.getvalue()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
