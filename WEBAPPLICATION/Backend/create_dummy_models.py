
import torch
import torch.nn as nn
import joblib
import numpy as np
from pathlib import Path
import os
import lightgbm as lgb

# Define dummy Autoformer class structure to allow loading
class DummyAutoformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(96, 96) # Input 24h, Output 24h features
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Return dummy output shape: [batch, horizon, features]
        # Assuming last dim is target (1)
        batch_size = x_enc.shape[0]
        # Hardcoded horizon 96 for STLF? Or 24?
        # STLF is 24 hours (step=15min -> 96 steps)
        # return torch.zeros(batch_size, 96, 1)
        return self.linear(x_enc[:, -96:, 0:1]) # Simple pass through

def create_dummy_models():
    """Create valid dummy models for development."""
    os.makedirs("models", exist_ok=True)
    
    # 1. Create Dummy PyTorch Model (Autoformer)
    model = DummyAutoformer()
    torch.save(model, "models/autoformer_stlf.pt")
    print(f"Created PyTorch model at models/autoformer_stlf.pt")
    
    # 2. Create Dummy LightGBM Model
    # We need a trained booster or sklearn wrapper
    X = np.random.rand(100, 20) # 20 features
    y = np.random.rand(100)
    lgb_model = lgb.LGBMRegressor(n_estimators=10)
    lgb_model.fit(X, y)
    
    joblib.dump(lgb_model, "models/lightgbm_stlf.pkl")
    print(f"Created LightGBM model at models/lightgbm_stlf.pkl")
    
    # 3. Create Dummy LTLF Model
    ltlf_model = lgb.LGBMRegressor(n_estimators=10)
    ltlf_model.fit(X, y)
    joblib.dump(ltlf_model, "models/ltlf_recursive.pkl")
    print(f"Created LTLF model at models/ltlf_recursive.pkl")

if __name__ == "__main__":
    create_dummy_models()
