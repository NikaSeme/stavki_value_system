# Repro Pack: How to Run Everything

## 1. Environment
- **Python:** 3.10+ (Verified in `python_version.txt`)
- **Dependencies:** See `requirements.txt`
- **OS:** macOS / Linux (Ubuntu 22.04 LTS recommended)

## 2. Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Configure keys manually
```

## 3. "Golden Commands" (Verified)

### A. Run Test Suite
```bash
python -m pytest tests/
```

### B. Train Models (Full Pipeline)
```bash
# 1. Train Base Models
python scripts/train_model.py
python scripts/train_neural_model.py
python scripts/train_poisson_model.py

# 2. Train Ensemble
python scripts/train_ensemble_model.py
```

### C. Live Inference (Scheduler)
```bash
python run_scheduler.py --telegram --interval 60
```

### D. Diagnostics & Metrics
```bash
python scripts/validate_metrics_comprehensive.py
```

### E. One-Off Pipeline Run
```bash
python -m src.cli run --matches data/processed/features.csv --odds data/odds.csv
```
