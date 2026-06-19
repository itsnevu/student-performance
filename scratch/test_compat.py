import os
import joblib
import pandas as pd
import numpy as np

PREPROCESSOR_FILE = "outputs/models/preprocessors.pkl"
MODELS_PATH = "outputs/models"

pre_objs = joblib.load(PREPROCESSOR_FILE)
ct = pre_objs['column_transformer']
scaler = pre_objs['scaler']
selector = pre_objs['selector']

columns = ['age', 'sex', 'graduated_h_school_type', 'scholarship_type', 'additional_work', 
           'activity', 'partner', 'total_salary', 'transport', 'accomodation', 'mother_ed', 
           'farther_ed', 'siblings', 'parental_status', 'mother_occup', 'father_occup', 
           'weekly_study_hours', 'reading_non_scientific', 'reading_scientific', 
           'attendance_seminars_dep', 'impact_of_projects', 'attendances_classes', 
           'preparation_midterm_company', 'preparation_midterm_time', 'taking_notes', 
           'listenning', 'discussion_improves_interest', 'flip_classrom', 'grade_previous', 
           'grade_expected', 'course_id']

dummy_data = {col: [2] for col in columns}
df = pd.DataFrame(dummy_data)

X_ct = ct.transform(df)
X_scaled = scaler.transform(X_ct)
X_selected = selector.transform(X_scaled)

for f in sorted(os.listdir(MODELS_PATH)):
    if f.endswith('.pkl') and f != 'preprocessors.pkl':
        try:
            model = joblib.load(os.path.join(MODELS_PATH, f))
            pred = model.predict(X_selected)
            print(f"[Compatible] {f} (predictions shape: {pred.shape})")
        except Exception as e:
            print(f"[Incompatible] {f}: {e}")
