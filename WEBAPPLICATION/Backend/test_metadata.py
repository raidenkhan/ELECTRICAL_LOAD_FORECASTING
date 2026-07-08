from app.ml.model_metadata import list_models, get_model_metadata

# Test listing models
models = list_models()
print(f"Registered models: {len(models)}")
for m in models:
    print(f"  - {m['name']} v{m['version']} ({m['type']})")

# Test getting specific metadata
meta = get_model_metadata("decomp_engine")
if meta:
    print(f"DecompEngine: MAE={meta.metrics.get('mae')}, RMSE={meta.metrics.get('rmse')}")
else:
    print("DecompEngine metadata not found")