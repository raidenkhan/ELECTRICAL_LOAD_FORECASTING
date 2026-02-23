import joblib
import os

model_path = r"c:\Users\User\Desktop\project\llm_from_scratch-main\LOADFORECASINGPROJECT\WEBAPPLICATION\Backend\models\lightgbm_stlf.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print(f"Model type: {type(model)}")
    if isinstance(model, dict):
        first_key = list(model.keys())[0]
        sub = model[first_key]
        if hasattr(sub, 'feature_name'):
            print(f"Sub-model features: {sub.feature_name()}")
        elif hasattr(sub, 'feature_names'):
            print(f"Sub-model features: {sub.feature_names}")
        else:
            print(f"Sub-model has no feature_name attribute. Dir: {dir(sub)}")
else:
    print("Model not found")
