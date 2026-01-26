#!/usr/bin/env python3
"""
Generate Production Fingerprint (Log)
records git sha, python version, and env info.
"""
import sys
import subprocess
import platform
from datetime import datetime
from pathlib import Path

def main():
    out_path = Path("audit_pack/RUN_LOGS/PROD_FINGERPRINT.log")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w') as f:
        f.write(f"STAVKI VALUE SYSTEM V5 FINGERPRINT\n")
        f.write(f"Generated: {datetime.utcnow()} UTC\n")
        f.write("-" * 40 + "\n")
        
        # Git
        try:
            sha = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('ascii')
            f.write(f"Git Commit: {sha}\n")
        except:
            f.write("Git Commit: Unknown (not a git repo?)\n")
            
        # System
        f.write(f"Python: {sys.version.split()[0]}\n")
        f.write(f"Platform: {platform.platform()}\n")
        f.write(f"CWD: {os.getcwd()}\n")
        
        # Models
        models_dir = Path("models")
        f.write("\nModels Present:\n")
        if models_dir.exists():
            for m in models_dir.glob("*"):
                f.write(f" - {m.name}\n")
        else:
            f.write(" - None\n")
            
    print(f"Fingerprint saved to {out_path}")
    
import os
if __name__ == "__main__":
    main()
