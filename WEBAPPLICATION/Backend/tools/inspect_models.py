
import torch
import joblib
import sys
import os

def inspect_models():
    print("Inspecting models...")
    
    # 1. Inspect LightGBM
    lgb_path = "models/lightgbm_stlf.pkl"
    if os.path.exists(lgb_path):
        print(f"\n--- LightGBM ({lgb_path}) ---")
        try:
            model = joblib.load(lgb_path)
            print(f"Type: {type(model)}")
            if isinstance(model, dict):
                print(f"Keys: {list(model.keys())[:5]} ...")
                first_key = list(model.keys())[0]
                print(f"Value[{first_key}] type: {type(model[first_key])}")
            else:
                print(f"Model keys/params: {dir(model)[:5]}")
        except Exception as e:
            print(f"Error loading LightGBM: {e}")
            
    # 2. Inspect Autoformer
    pt_path = "models/autoformer_stlf.pt"
    if os.path.exists(pt_path):
        print(f"\n--- Autoformer ({pt_path}) ---")
        try:
            state_dict = torch.load(pt_path, map_location='cpu')
            print(f"Type: {type(state_dict)}")
            if isinstance(state_dict, dict):
                print("Detected State Dictionary.")
                print("Extracting Architecture Params from weights:")
                
                # Input Dim from enc_embedding.weight [d_model, input_dim]
                if "enc_embedding.weight" in state_dict:
                    w = state_dict["enc_embedding.weight"]
                    print(f"enc_embedding.weight: {w.shape} => d_model={w.shape[0]}, input_dim={w.shape[1]}")
                    
                # Horizon/Seq_len from fc_seasonal.weight [horizon, seq_len * d_model]
                if "fc_seasonal.weight" in state_dict:
                    w = state_dict["fc_seasonal.weight"]
                    print(f"fc_seasonal.weight:   {w.shape} => horizon={w.shape[0]}")
                    # We can't deduce seq_len easily without d_model, but we have d_model from above
                    
                # Heads/Layers are harder, usually hidden
                # encoder.layers.0 ...
                keys = list(state_dict.keys())
                layers = [k for k in keys if "encoder.layers" in k]
                if layers:
                    max_layer = max([int(k.split('.')[2]) for k in layers if k.split('.')[2].isdigit()])
                    print(f"Estimated e_layers: {max_layer + 1}")
                    
            elif isinstance(state_dict, torch.nn.Module):
                print("Is nn.Module")
                
        except Exception as e:
            print(f"Error loading Autoformer: {e}")

if __name__ == "__main__":
    inspect_models()
