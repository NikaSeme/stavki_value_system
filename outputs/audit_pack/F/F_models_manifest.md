# Models Manifest

## 1. Active Models (PLAN 1 Verification)
We have successfully implemented the **Ensemble Strategy** outlined in PLAN 1.

### ✅ Model A: Poisson Statistical
- **Path:** `models/poisson_v1_*.pkl`
- **Role:** Baseline probability estimation based on team goal scoring rates.

### ✅ Model B: CatBoost (ML)
- **Path:** `models/catboost_v1_*.pkl`
- **Role:** Primary ML predictor using 20+ features (Form, Elo, Market).
- **Performance:** Best individual accuracy (~59%).

### ✅ Model C: Neural Network
- **Path:** `models/neural_v1_*.pt`
- **Role:** Deep learning component for ensemble diversity.

### ✅ Meta-Model (Stacking)
- **Path:** `models/meta_model_v1_*.pkl`
- **Role:** Combines A+B+C outputs. Calibrated via Isotonic Regression.

## 2. Artifacts
See `models_manifest.json` for precise file paths and version hashes.
