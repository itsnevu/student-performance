import joblib
import pandas as pd
import numpy as np

PREPROCESSOR_FILE = "outputs/models/preprocessors.pkl"
MODEL_FILE = "outputs/models/random_forest.pkl"

pre_objs = joblib.load(PREPROCESSOR_FILE)
model = joblib.load(MODEL_FILE)

ct = pre_objs['column_transformer']
scaler = pre_objs['scaler']
selector = pre_objs['selector']

# Create a single dummy input DataFrame with matching columns
columns = ['age', 'sex', 'graduated_h_school_type', 'scholarship_type', 'additional_work', 
           'activity', 'partner', 'total_salary', 'transport', 'accomodation', 'mother_ed', 
           'farther_ed', 'siblings', 'parental_status', 'mother_occup', 'father_occup', 
           'weekly_study_hours', 'reading_non_scientific', 'reading_scientific', 
           'attendance_seminars_dep', 'impact_of_projects', 'attendances_classes', 
           'preparation_midterm_company', 'preparation_midterm_time', 'taking_notes', 
           'listenning', 'discussion_improves_interest', 'flip_classrom', 'grade_previous', 
           'grade_expected', 'course_id']

# Populate with dummy values (integers)
dummy_data = {col: [2] for col in columns}
df = pd.DataFrame(dummy_data)

# Transform
X_ct = ct.transform(df)
X_scaled = scaler.transform(X_ct)
X_selected = selector.transform(X_scaled)

# Predict
pred = model.predict(X_selected)
prob = model.predict_proba(X_selected) if hasattr(model, 'predict_proba') else None

print("Prediction:", pred)
print("Probabilities:", prob)
