#!/usr/bin/env python3
"""
Train CatBoost Model (v3)
Trains on strict time-split dataset (train_v3.parquet).
Outputs: model, scaler, calibrator, metrics, plots.
"""
import argparse
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier, Pool

# Config
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
AUDIT_METRICS_DIR = Path("audit_pack/A6_metrics")
AUDIT_METRICS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data/processed")

def load_data():
    print("Loading datasets...")
    train = pd.read_parquet(DATA_DIR / "train_v3.parquet")
    val = pd.read_parquet(DATA_DIR / "val_v3.parquet")
    test = pd.read_parquet(DATA_DIR / "test_v3.parquet")
    
    # Feature Engineering / Selection
    # Drop non-features and ensure numeric
    exclude_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'match_id', 'id', 'Season', 'Div', 'Referee']
    
    # Select only numeric columns
    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    print(f"Selected {len(feature_cols)} features: {feature_cols[:5]}...")
    
    # Simple target mapping: H=0, D=1, A=2
    target_map = {'H': 0, 'D': 1, 'A': 2}
    
    def prepare(df):
        X = df[feature_cols].copy()
        # Handle NaNs: Fill with mean or median or 0? 0 is safe for StandardScaler (mean centering)
        # But better: median imputation from train set. 
        # For simplicity in this script: fill 0 (Scaler will center it)
        X = X.fillna(0) 
        y = df['FTR'].map(target_map)
        return X, y

    X_train, y_train = prepare(train)
    X_val, y_val = prepare(val)
    X_test, y_test = prepare(test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols

def train_model(X_train, y_train, X_val, y_val):
    print("Training CatBoost...")
    
    # Scale features (Good practice even for trees if we calibrate later)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Initialize CatBoost
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function='MultiClass',
        eval_metric='MultiClass',
        random_seed=42,
        verbose=100,
        allow_writing_files=False
    )
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=(X_val_scaled, y_val),
        early_stopping_rounds=50
    )
    
    return model, scaler

def calibrate_model(model, scaler, X_val, y_val):
    print("Calibrating (Isotonic)...")
    # Wrap with CalibratedClassifierCV
    # Note: CCCV usually needs a base estimator.
    # We can use 'prefit' if we trust the val set, or standard CV.
    # Given strict time split, we should use the Validation set for calibration.
    
    X_val_scaled = scaler.transform(X_val)
    
    calibrator = CalibratedClassifierCV(model, method='isotonic', cv='prefit', ensemble=False)
    calibrator.fit(X_val_scaled, y_val)
    
    return calibrator

def evaluate(calibrator, scaler, X_test, y_test, feature_names):
    print("Evaluating on Test Set...")
    X_test_scaled = scaler.transform(X_test)
    probs = calibrator.predict_proba(X_test_scaled)
    preds = calibrator.predict(X_test_scaled)
    
    # Metrics
    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, probs)
    
    # Brier (multiclass brier is sum of Brier per class / K or similar, SKLearn doesn't have direct MultiClass Brier)
    # compute per class
    y_test_onehot = pd.get_dummies(y_test).values
    # Ensure columns 0,1,2 exist
    
    brier = np.mean(np.sum((probs - y_test_onehot)**2, axis=1))
    
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test LogLoss:  {ll:.4f}")
    print(f"Test Brier:    {brier:.4f}")
    
    metrics = {
        "accuracy": acc,
        "log_loss": ll,
        "brier_score": brier
    }
    
    # Plots
    # Reliability Curve (Per Class)
    plt.figure(figsize=(10, 6))
    for i, label in enumerate(['Home', 'Draw', 'Away']):
        prob_true, prob_pred = calibration_curve(y_test == i, probs[:, i], n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=label)
    
    plt.plot([0, 1], [0, 1], 'k--', label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Reliability Curve (Test Set)")
    plt.legend()
    plt.grid(True)
    plt.savefig(AUDIT_METRICS_DIR / "calibration_plot_catboost.png")
    plt.close()
    
    return metrics

def main():
    # 1. Load
    X_train, y_train, X_val, y_val, X_test, y_test, features = load_data()
    
    # 2. Train
    model, scaler = train_model(X_train, y_train, X_val, y_val)
    
    # 3. Calibrate
    calibrator = calibrate_model(model, scaler, X_val, y_val)
    
    # 4. Evaluate
    metrics = evaluate(calibrator, scaler, X_test, y_test, features)
    
    # 5. Save Artifacts
    print("Saving artifacts...")
    
    # Save Metrics
    with open(AUDIT_METRICS_DIR / "catboost_metrics.json", "w") as f:
        json.dump({
            "metrics": metrics,
            "features": features,
            "train_rows": len(X_train),
            "val_rows": len(X_val),
            "test_rows": len(X_test)
        }, f, indent=2)
        
    # Save Models with Hash (v3 + latest symlink)
    def save_artifact(obj, name_v3, name_latest):
        path_v3 = MODELS_DIR / name_v3
        path_latest = MODELS_DIR / name_latest
        joblib.dump(obj, path_v3)
        # Create symlink/copy for 'latest' loading
        # Actually standard copy to avoid symlink issues on some OS
        import shutil
        shutil.copy(path_v3, path_latest)
        print(f"Saved {path_v3} -> {path_latest}")

    save_artifact(model, "catboost_v3.pkl", "catboost_v1_latest.pkl") # Use v1_latest name for compatibility? 
    # Wait, loader looks for 'catboost_v1_latest.pkl'. 
    # I should probably update loader to look for v3 or just overwrite v1_latest. 
    # Re-using v1_latest to avoid breaking loader if I revert, but better to be explicit.
    # Loader code: model_file = self.models_dir / 'catboost_v1_latest.pkl'
    # I will overwrite catboost_v1_latest.pkl to immediately act as the deployed model.
    
    save_artifact(scaler, "scaler_v3.pkl", "scaler_v1_latest.pkl")
    save_artifact(calibrator, "calibrator_v3.pkl", "calibrator_v1_latest.pkl")
    
    # Metadata
    meta = {
        "version": "v3.2",
        "train_date": datetime.now().isoformat(),
        "num_features": len(features),
        "features": features,
        "metrics": {"test": metrics}
    }
    with open(MODELS_DIR / "metadata_v3.json", "w") as f:
        json.dump(meta, f, indent=2)
    import shutil
    shutil.copy(MODELS_DIR / "metadata_v3.json", MODELS_DIR / "metadata_v1_latest.json")
    
    print("✅ Training Complete.")

from datetime import datetime
if __name__ == "__main__":
    main()
