import torch
import os

model_path = r"c:\Users\User\Desktop\project\llm_from_scratch-main\LOADFORECASINGPROJECT\WEBAPPLICATION\Backend\models\autoformer_stlf.pt"
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            keys = checkpoint["state_dict"].keys()
        else:
            keys = checkpoint.keys()
        print("Model Keys found:")
        for k in list(keys)[:20]:
            print(f"  {k}")
        
        # Check for specific layers
        for layer in ["future_proj", "future_refinement", "revin"]:
            found = any(layer in k for k in keys)
            print(f"Layer '{layer}' found: {found}")
    else:
        print(f"Checkpoint is type {type(checkpoint)}")
else:
    print("Model not found")
