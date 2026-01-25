import os
import json
import hashlib
from pathlib import Path

# Configuration
MODELS_DIR = 'models'
OUTPUT_DIR = 'audit_pack/A5_models'
MANIFEST_FILE = os.path.join(OUTPUT_DIR, 'models_manifest.json')

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Discovering Models in {MODELS_DIR}...")

def calculate_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

models = []
if os.path.exists(MODELS_DIR):
    for root, dirs, files in os.walk(MODELS_DIR):
        for file in files:
            if file.endswith(('.pkl', '.joblib', '.pt', '.h5', '.json')):
                full_path = os.path.join(root, file)
                stat = os.stat(full_path)
                
                model_type = "unknown"
                if "catboost" in file: model_type = "catboost"
                elif "poisson" in file: model_type = "poisson"
                elif "neural" in file or ".pt" in file: model_type = "neural_network"
                elif "ensemble" in file: model_type = "ensemble"
                elif "scaler" in file: model_type = "scaler"
                
                models.append({
                    "filename": file,
                    "path": full_path,
                    "type": model_type,
                    "size_bytes": stat.st_size,
                    "hash_sha256": calculate_sha256(full_path),
                    "modified_time": stat.st_mtime
                })
else:
    print(f"WARNING: {MODELS_DIR} does not exist!")

with open(MANIFEST_FILE, 'w') as f:
    json.dump({"models": models, "count": len(models)}, f, indent=2)

print(f"Found {len(models)} model artifacts.")
print(f"Saved manifest to {MANIFEST_FILE}")
