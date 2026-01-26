#!/usr/bin/env python3
"""
Generate Models Manifest (A5)
Hashes model artifacts and generates manifest JSON.
"""
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime

MODELS_DIR = Path("models")
MANIFEST_FILE = Path("audit_pack/A5_models/models_manifest.json")
MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)

def get_hash(file_path):
    if not file_path.exists():
        return None
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    print("Generating Models Manifest...")
    
    artifacts = [
        "catboost_v3.pkl",
        "catboost_v1_latest.pkl", # Deployed
        "scaler_v3.pkl",
        "scaler_v1_latest.pkl",
        "calibrator_v3.pkl",
        "calibrator_v1_latest.pkl",
        "metadata_v3.json"
    ]
    
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "artifacts": {}
    }
    
    for name in artifacts:
        path = MODELS_DIR / name
        file_hash = get_hash(path)
        manifest["artifacts"][name] = {
            "path": str(path),
            "sha256": file_hash,
            "exists": file_hash is not None
        }
        print(f"  {name}: {'✅' if file_hash else '❌'}")
        
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"✅ Manifest saved to {MANIFEST_FILE}")

if __name__ == "__main__":
    main()
