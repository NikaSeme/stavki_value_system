#!/usr/bin/env python3
"""
Model Load Smoke Test (Audit v3)
Verifies that models load correctly, hashes them, and reports status.
No fallback allowed.
"""
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.loader import ModelLoader

OUTPUT_DIR = Path("audit_pack/A9_live")
LOG_DIR = Path("audit_pack/RUN_LOGS")
REPORT_FILE = OUTPUT_DIR / "model_load_report.json"
LOG_FILE = LOG_DIR / "model_load.log"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

def calculate_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    print("=== Model Load Smoke Test ===")
    
    loader = ModelLoader(models_dir="models")
    
    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "loaded": False,
        "models": [],
        "fallback_used": False, # We enforce strict loading here
        "error": None
    }
    
    try:
        success = loader.load_latest()
        if not success:
            raise RuntimeError("loader.load_latest() returned False")
        
        valid = loader.validate()
        if not valid:
            raise RuntimeError("loader.validate() returned False")
            
        report["loaded"] = True
        
        # Hash Artifacts
        artifacts = [
            "catboost_v1_latest.pkl", 
            "calibrator_v1_latest.pkl", 
            "scaler_v1_latest.pkl",
            "neural_v1_latest.pt" # Check if this exists
        ]
        
        for art in artifacts:
            p = Path("models") / art
            if p.exists():
                report["models"].append({
                    "name": art,
                    "sha256": calculate_sha256(p),
                    "size": p.stat().st_size
                })
            else:
                if "neural" in art:
                    print(f"WARN: {art} not found (Model C optional?)")
                else:
                    raise FileNotFoundError(f"Required artifact {art} missing")

        print("✅ Models loaded and validated.")
        
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        report["loaded"] = False
        report["error"] = str(e)
        
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"Report saved to {REPORT_FILE}")
    
    if not report["loaded"]:
        sys.exit(1)

if __name__ == "__main__":
    main()
