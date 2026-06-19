import joblib
import os

PREPROCESSOR_FILE = "outputs/models/preprocessors.pkl"
if os.path.exists(PREPROCESSOR_FILE):
    pre_objs = joblib.load(PREPROCESSOR_FILE)
    print("Keys in preprocessors.pkl:", pre_objs.keys())
    
    ct = pre_objs.get('column_transformer')
    if ct:
        print("\nColumn Transformer Transformers:")
        for name, transformer, columns in ct.transformers_:
            print(f"- {name}: {columns}")
        if hasattr(ct, 'feature_names_in_'):
            print("\nFeature names expected (feature_names_in_):")
            print(list(ct.feature_names_in_))
            print("Total features:", len(ct.feature_names_in_))
            
    scaler = pre_objs.get('scaler')
    if scaler:
        print(f"\nScaler: {scaler}")
        
    selector = pre_objs.get('selector')
    if selector:
        print(f"\nSelector: {selector}")
        if hasattr(selector, 'get_support'):
            print("Selected features mask sum:", selector.get_support().sum())
else:
    print("Preprocessor file not found!")
