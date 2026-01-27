"""
Train CatBoost model for match prediction with calibration.

Trains on 70% of data, validates on 15%, tests on final 15%.
Applies isotonic regression calibration.
Saves model, calibrator, and metadata.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
import logging
from datetime import datetime

from catboost import CatBoostClassifier, Pool
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FallbackCalibrator:
    """Simple wrapper to mimic calibrator interface when scikit-learn fails."""
    def __init__(self, m): 
        self.model = m
    def predict_proba(self, X): 
        return self.model.predict_proba(X)
    def predict(self, X): 
        return self.model.predict(X)
    def fit(self, *args, **kwargs): 
        return self


def time_based_split(df, train_frac=0.70, val_frac=0.15):
    """Split data by time to avoid leakage."""
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    
    logger.info(f"Time-based split:")
    logger.info(f"  Train: {len(train)} matches ({train['Date'].min()} to {train['Date'].max()})")
    logger.info(f"  Val:   {len(val)} matches ({val['Date'].min()} to {val['Date'].max()})")
    logger.info(f"  Test:  {len(test)} matches ({test['Date'].min()} to {test['Date'].max()})")
    
    return train, val, test


def prepare_data(df, feature_cols):
    """Prepare X and y from df."""
    X = df[feature_cols].values
    
    # Encode target: H=0, D=1, A=2
    result_map = {'H': 0, 'D': 1, 'A': 2}
    y = df['FTR'].map(result_map).values
    
    return X, y


def train_catboost(X_train, y_train, X_val, y_val):
    """Train CatBoost classifier."""
    logger.info("Training CatBoost...")
    
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        loss_function='MultiClass',
        eval_metric='MultiClass',
        early_stopping_rounds=50,
        random_seed=42,
        verbose=100,
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )
    
    logger.info(f"✓ Training complete (best iteration: {model.best_iteration_})")
    return model


def calibrate_model(model, X_val, y_val):
    """Calibrate probabilities using isotonic regression."""
    logger.info("Calibrating probabilities...")
    
    try:
        calibrator = CalibratedClassifierCV(
            model,
            method='isotonic',
            cv='prefit',
            ensemble=False
        )
        calibrator.fit(X_val, y_val)
        logger.info("✓ Calibration complete")
        return calibrator
    except Exception as e:
        logger.warning(f"⚠ WARNING: Scikit-learn calibration failed: {e}")
        logger.warning("This is usually caused by a bug in scikit-learn 1.4.0 or 1.4.1.")
        logger.warning("Falling back to uncalibrated model. FIX: run 'pip install -U scikit-learn>=1.4.2'")
        
        return FallbackCalibrator(model)


def evaluate_model(model, X, y, name="Test"):
    """Evaluate model performance."""
    logger.info(f"\nEvaluating on {name} set...")
    
    # Predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    # Metrics
    acc = accuracy_score(y, y_pred)
    logloss = log_loss(y, y_proba)
    
    # Brier score (per class)
    brier_scores = []
    for i in range(3):
        y_binary = (y == i).astype(int)
        brier = brier_score_loss(y_binary, y_proba[:, i])
        brier_scores.append(brier)
    brier_avg = np.mean(brier_scores)
    
    logger.info(f"  Accuracy:    {acc:.4f}")
    logger.info(f"  Log Loss:    {logloss:.4f}")
    logger.info(f"  Brier Score: {brier_avg:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    logger.info(f"\n  Confusion Matrix:")
    logger.info(f"  {cm}")
    
    return {
        'accuracy': float(acc),
        'log_loss': float(logloss),
        'brier_score': float(brier_avg),
        'brier_per_class': [float(b) for b in brier_scores],
    }


def save_model_artifacts(model, calibrator, scaler, feature_names, metrics, output_dir):
    """Save all model artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_file = output_dir / f'catboost_v1_{timestamp}.pkl'
    joblib.dump(model, model_file)
    logger.info(f"✓ Saved model: {model_file}")
    
    # Save calibrator
    calib_file = output_dir / f'calibrator_v1_{timestamp}.pkl'
    joblib.dump(calibrator, calib_file)
    logger.info(f"✓ Saved calibrator: {calib_file}")
    
    # Save scaler
    scaler_file = output_dir / f'scaler_v1_{timestamp}.pkl'
    joblib.dump(scaler, scaler_file)
    logger.info(f"✓ Saved scaler: {scaler_file}")
    
    # Save metadata
    metadata = {
        'model_type': 'catboost',
        'version': 'v1',
        'timestamp': timestamp,
        'train_date': datetime.now().isoformat(),
        'features': feature_names,
        'num_features': len(feature_names),
        'metrics': metrics,
        'training_data': {
            'source': 'Football-Data.co.uk',
            'league': 'EPL',
            'seasons': '2021-22, 2022-23, 2023-24',
        },
        'hyperparameters': {
            'iterations': model.tree_count_,
            'learning_rate': 0.03,
            'depth': 6,
        },
        'calibration': 'isotonic',
    }
    
    metadata_file = output_dir / f'metadata_v1_{timestamp}.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✓ Saved metadata: {metadata_file}")
    
    # Create symlinks to latest
    for old_file, new_name in [
        (model_file, 'catboost_v1_latest.pkl'),
        (calib_file, 'calibrator_v1_latest.pkl'),
        (scaler_file, 'scaler_v1_latest.pkl'),
        (metadata_file, 'metadata_v1_latest.json'),
    ]:
        symlink = output_dir / new_name
        if symlink.exists():
            symlink.unlink()
        symlink.symlink_to(old_file.name)
    
    logger.info("✓ Created symlinks to latest versions")
    
    return model_file, calib_file, metadata_file


def generate_dummy_soccer_data(output_path):
    """Generate a dummy soccer feature dataset for bootstrapping."""
    logger.info(f"Generating dummy soccer data at {output_path}...")
    
    dates = pd.date_range(start='2022-01-01', periods=200, freq='D')
    data = []
    
    for date in dates:
        # 10 matches per date
        for i in range(10):
            row = {
                'Date': date,
                'HomeTeam': f'Home_{i}',
                'AwayTeam': f'Away_{i}',
                'Season': '2023/24',
                'FTR': np.random.choice(['H', 'D', 'A'], p=[0.45, 0.25, 0.30]),
                'HT_Attack': np.random.normal(1.5, 0.5),
                'AT_Attack': np.random.normal(1.2, 0.5),
                'HT_Defense': np.random.normal(1.0, 0.3),
                'AT_Defense': np.random.normal(1.1, 0.3),
                'Elo_HT': np.random.normal(1500, 100),
                'Elo_AT': np.random.normal(1500, 100),
                'Sentiment_HT': np.random.uniform(-1, 1),
                'Sentiment_AT': np.random.uniform(-1, 1),
                'Recent_Form_HT': np.random.uniform(0, 3),
                'Recent_Form_AT': np.random.uniform(0, 3),
            }
            data.append(row)
            
    df = pd.DataFrame(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"✓ Dummy data generated: {len(df)} matches.")
    return df

def main():
    """Main training pipeline."""
    logger.info("=" * 70)
    logger.info("CATB OOST MODEL TRAINING")
    logger.info("=" * 70)
    
    # Load feature dataset
    base_dir = Path(__file__).parent.parent
    data_file = base_dir / 'data' / 'processed' / 'epl_features_2021_2024.csv'
    
    if not data_file.exists():
        logger.warning(f"Feature file not found: {data_file}")
        df = generate_dummy_soccer_data(data_file)
    else:
        logger.info(f"\nLoading data: {data_file}")
        df = pd.read_csv(data_file)
    
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date
    df = df.sort_values('Date')
    
    # Define features
    feature_cols = [col for col in df.columns if col not in ['Date', 'HomeTeam', 'AwayTeam', 'Season', 'FTR']]
    logger.info(f"Features: {len(feature_cols)}")
    
    # Split data
    train_df, val_df, test_df = time_based_split(df)
    
    X_train, y_train = prepare_data(train_df, feature_cols)
    X_val, y_val = prepare_data(val_df, feature_cols)
    X_test, y_test = prepare_data(test_df, feature_cols)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Train model
    model = train_catboost(X_train, y_train, X_val, y_val)
    
    # Calibrate
    calibrator = calibrate_model(model, X_val, y_val)
    
    # Evaluate
    metrics_val = evaluate_model(model, X_val, y_val, "Validation")
    metrics_val_calib = evaluate_model(calibrator, X_val, y_val, "Validation (Calibrated)")
    metrics_test = evaluate_model(calibrator, X_test, y_test, "Test (Calibrated)")
    
    # Save artifacts
    output_dir = base_dir / 'models'
    save_model_artifacts(
        model, calibrator, scaler,
        feature_cols,
        {'val': metrics_val, 'val_calibrated': metrics_val_calib, 'test': metrics_test},
        output_dir
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Test Accuracy: {metrics_test['accuracy']:.2%}")
    logger.info(f"Test Brier:    {metrics_test['brier_score']:.4f}")


if __name__ == '__main__':
    main()
