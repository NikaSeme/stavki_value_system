#!/usr/bin/env python3
"""
Generate Models Manifest (V5)
Hashes all models in models/ and creates models_manifest.json
"""
import hashlib
import json
import os
from pathlib import Path

def sha256_checksum(filename, block_size=65536):
    sha256 = hashlib.sha256()
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            sha256.update(block)
    return sha256.hexdigest()

def main():
    models_dir = Path("models")
    manifest = {}
    
    if not models_dir.exists():
        print("No models directory found.")
        return

    print("Hashing models...")
    for f in models_dir.glob("*"):
        if f.is_file() and f.name != ".DS_Store":
             # Skip manifest itself if it exists there
             if "manifest" in f.name: continue
             
             h = sha256_checksum(f)
             manifest[f.name] = {
                 "sha256": h,
                 "size": f.stat().st_size,
                 "path": str(f)
             }
             print(f"  {f.name}: {h[:8]}...")
             
    out_path = Path("audit_pack/A5_models/models_manifest.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Manifest saved to {out_path}")

if __name__ == "__main__":
    main()
