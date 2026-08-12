# 🎓 Student Performance Prediction — ML Pipeline & Prediction App

An end-to-end machine learning system that predicts student graduation outcomes based on demographic and academic factors. The project includes a modular ML audit pipeline, a REST API for inference, and a web interface for trying out predictions directly.

---

## ✨ Features

Full ML pipeline covering data loading, EDA, cleaning, and feature engineering through to model evaluation. Automated comparison of 10+ algorithms (Random Forest, XGBoost, SVM, KNN, and more) with hyperparameter tuning. A REST API (FastAPI) for serving the model to other applications. A web frontend for entering student data and viewing predictions. Interactive notebooks for exploring the pipeline step by step. Ready to run with Docker.

## 🧱 Project Structure

```
student-performance/
├── data/raw/          # Raw dataset (CSV)
├── notebooks/         # Interactive audit pipeline notebooks
├── src/               # ML pipeline modules
├── frontend/          # Web app (Next.js) for input & predictions
├── outputs/           # Trained models (.pkl) & evaluation results
├── app.py             # FastAPI backend for inference
├── main.py            # Entry point for the automated audit pipeline
├── Dockerfile
└── requirements.txt
```

## 🧠 Methodology

The pipeline follows five ML audit phases. Phase 1, Data Loading & EDA, profiles the data and target distribution. Phase 2, Data Preparation, covers cleaning, imputation, transformation, outlier handling, splitting, normalization, and imbalance handling. Phase 3, Feature Engineering, covers feature addition, extraction, reduction, selection, and PCA. Phase 4, Modeling, compares ten models with hyperparameter tuning. Phase 5, Evaluation, measures F1, Accuracy, Precision, Recall, AUC-ROC, and cross-validation mean/std.

## 🛠️ Tech Stack

Machine Learning: Python, Pandas, NumPy, Scikit-Learn, XGBoost, Imbalanced-learn (SMOTE), Joblib.
Backend API: FastAPI, Uvicorn.
Frontend: TypeScript (Next.js).
Deployment: Docker, Vercel.

## 🚀 Getting Started

### Set up the environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run the audit pipeline

Place your CSV dataset in `data/raw/`, then run:

```bash
python main.py
```

Trained models and evaluation charts are saved automatically to the `outputs/` folder.

### Run the API

```bash
uvicorn app:app --reload
```

### Run the frontend

```bash
cd frontend
npm install
npm run dev
```

### Explore interactively

Open the notebooks in the `notebooks/` folder to walk through each phase of the pipeline in detail.

## 📊 Dataset

Uses a student performance dataset with 31 features, covering demographics (age, gender, family background), study habits, and academic history.

## 📄 License

Not yet specified.
