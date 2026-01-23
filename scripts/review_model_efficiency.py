"""
T360: Model Efficiency Review Script

Comprehensive analysis of model performance to determine
if further optimization is justified or if models have
reached optimal efficiency.
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import models
from scripts.train_poisson_model import PoissonMatchPredictor
from src.models.loader import ModelLoader
import torch
from scripts.train_neural_model import DenseNN


def load_test_data():
    """Load test dataset for evaluation."""
    features_df = pd.read_csv('data/processed/epl_features_2021_2024.csv')
    features_df['Date'] = pd.to_datetime(features_df['Date'])
    features_df = features_df.sort_values('Date')
    
    # Use last 15% as test (same as model training)
    n = len(features_df)
    test_start = int(n * 0.85)
    test_df = features_df.iloc[test_start:]
    
    return test_df


def evaluate_poisson_model(test_df):
    """Evaluate Poisson model (Model A)."""
    print("\n" + "="*70)
    print("MODEL A (Poisson) EVALUATION")
    print("="*70)
    
    model = PoissonMatchPredictor.load('models/poisson_v1_latest.pkl')
    
    # Prepare data
    df_poisson = test_df.copy()
    if 'HomeTeam' not in df_poisson.columns:
        df_poisson = df_poisson.rename(columns={'home_team': 'HomeTeam', 'away_team': 'AwayTeam'})
    
    # Predict
    predictions = model.predict(df_poisson)
    probs = predictions[['prob_home', 'prob_draw', 'prob_away']].values
    
    # Get true labels
    result_map = {'H': 0, 'D': 1, 'A': 2}
    y_true = test_df['FTR'].map(result_map).values[:len(probs)]
    
    # Calculate metrics
    from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
    
    # Accuracy
    y_pred = np.argmax(probs, axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    
    # Log loss
    logloss = log_loss(y_true, probs)
    
    # Brier score (average across outcomes)
    brier_scores = []
    for i in range(3):
        y_binary = (y_true == i).astype(int)
        brier = brier_score_loss(y_binary, probs[:, i])
        brier_scores.append(brier)
    brier_avg = np.mean(brier_scores)
    
    # ECE (Expected Calibration Error)
    ece = calculate_ece(y_true, probs)
    
    results = {
        'model': 'Poisson',
        'accuracy': float(accuracy),
        'log_loss': float(logloss),
        'brier_score': float(brier_avg),
        'ece': float(ece),
        'test_samples': len(y_true)
    }
    
    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Log Loss: {logloss:.4f}")
    print(f"  Brier Score: {brier_avg:.4f}")
    print(f"  ECE: {ece:.4f}")
    print(f"  Test Samples: {len(y_true)}")
    
    return results, probs


def evaluate_catboost_model(test_df):
    """Evaluate CatBoost model (Model B)."""
    print("\n" + "="*70)
    print("MODEL B (CatBoost) EVALUATION")
    print("="*70)
    
    loader = ModelLoader()
    loader.load_latest()
    
    # Get features
    feature_cols = [col for col in test_df.columns 
                   if col not in ['Date', 'HomeTeam', 'AwayTeam', 'Season', 'FTR']]
    X = test_df[feature_cols].values
    
    # Predict
    probs = loader.predict(X)
    
    # Get true labels
    result_map = {'H': 0, 'D': 1, 'A': 2}
    y_true = test_df['FTR'].map(result_map).values[:len(probs)]
    
    # Calculate metrics
    from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
    
    y_pred = np.argmax(probs, axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    logloss = log_loss(y_true, probs)
    
    brier_scores = []
    for i in range(3):
        y_binary = (y_true == i).astype(int)
        brier = brier_score_loss(y_binary, probs[:, i])
        brier_scores.append(brier)
    brier_avg = np.mean(brier_scores)
    
    ece = calculate_ece(y_true, probs)
    
    results = {
        'model': 'CatBoost',
        'accuracy': float(accuracy),
        'log_loss': float(logloss),
        'brier_score': float(brier_avg),
        'ece': float(ece),
        'test_samples': len(y_true)
    }
    
    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Log Loss: {logloss:.4f}")
    print(f"  Brier Score: {brier_avg:.4f}")
    print(f"  ECE: {ece:.4f}")
    
    return results, probs


def evaluate_neural_model(test_df):
    """Evaluate Neural model (Model C)."""
    print("\n" + "="*70)
    print("MODEL C (Neural Network) EVALUATION")
    print("="*70)
    
    # Load model
    checkpoint = torch.load('models/neural_v1_latest.pt', weights_only=False)
    
    model = DenseNN(input_dim=22, hidden_dims=[64, 32, 16], dropout=0.3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler = checkpoint['scaler']
    calibrators = checkpoint['calibrators']
    
    # Get features
    feature_cols = [col for col in test_df.columns 
                   if col not in ['Date', 'HomeTeam', 'AwayTeam', 'Season', 'FTR']]
    X = test_df[feature_cols].values
    
    # Scale and predict
    X_scaled = scaler.transform(X)
    X_tensor = torch.FloatTensor(X_scaled)
    
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).numpy()
    
    # Apply calibration
    if calibrators:
        calibrated = np.zeros_like(probs)
        for i, cal in enumerate(calibrators):
            calibrated[:, i] = cal.predict(probs[:, i])
        
        row_sums = calibrated.sum(axis=1, keepdims=True)
        probs = calibrated / row_sums
    
    # Get true labels
    result_map = {'H': 0, 'D': 1, 'A': 2}
    y_true = test_df['FTR'].map(result_map).values[:len(probs)]
    
    # Calculate metrics
    from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
    
    y_pred = np.argmax(probs, axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    logloss = log_loss(y_true, probs)
    
    brier_scores = []
    for i in range(3):
        y_binary = (y_true == i).astype(int)
        brier = brier_score_loss(y_binary, probs[:, i])
        brier_scores.append(brier)
    brier_avg = np.mean(brier_scores)
    
    ece = calculate_ece(y_true, probs)
    
    results = {
        'model': 'Neural',
        'accuracy': float(accuracy),
        'log_loss': float(logloss),
        'brier_score': float(brier_avg),
        'ece': float(ece),
        'test_samples': len(y_true)
    }
    
    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Log Loss: {logloss:.4f}")
    print(f"  Brier Score: {brier_avg:.4f}")
    print(f"  ECE: {ece:.4f}")
    
    return results, probs


def calculate_ece(y_true, y_prob, n_bins=10):
    """Calculate Expected Calibration Error."""
    y_pred = np.argmax(y_prob, axis=1)
    confidences = np.max(y_prob, axis=1)
    accuracies = (y_pred == y_true).astype(float)
    
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def evaluate_meta_model():
    """Evaluate meta-model ensemble."""
    print("\n" + "="*70)
    print("META-MODEL (Ensemble) EVALUATION")
    print("="*70)
    
    # Load meta-model metadata
    import json
    with open('models/meta_metadata_v1_latest.json') as f:
        meta_info = json.load(f)
    
    metrics = meta_info.get('metrics', {})
    
    print(f"\nMeta-Model Performance:")
    print(f"  Test Accuracy: {metrics.get('meta_test', {}).get('accuracy', 0):.2%}")
    print(f"  Test Brier: {metrics.get('meta_test', {}).get('brier_score', 0):.4f}")
    print(f"  Test Log Loss: {metrics.get('meta_test', {}).get('log_loss', 0):.4f}")
    
    print(f"\nBaseline (Simple Average):")
    print(f"  Accuracy: {metrics.get('baseline', {}).get('accuracy', 0):.2%}")
    print(f"  Brier: {metrics.get('baseline', {}).get('brier_score', 0):.4f}")
    
    return metrics


def main():
    """Run comprehensive model efficiency review."""
    print("="*70)
    print("T360: MODEL EFFICIENCY REVIEW")
    print("="*70)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    
    # Load test data
    print("\nLoading test dataset...")
    test_df = load_test_data ()
    print(f"Test samples: {len(test_df)}")
    
    # Evaluate each model
    poisson_results, poisson_probs = evaluate_poisson_model(test_df)
    catboost_results, catboost_probs = evaluate_catboost_model(test_df)
    neural_results, neural_probs = evaluate_neural_model(test_df)
    
    # Evaluate meta-model
    meta_metrics = evaluate_meta_model()
    
    # Compile all results
    all_results = {
        'poisson': poisson_results,
        'catboost': catboost_results,
        'neural': neural_results,
        'meta_model': meta_metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results
    output_dir = Path('outputs/diagnostics/T360')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'model_efficiency_review.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create comparison CSV
    df_results = pd.DataFrame([poisson_results, catboost_results, neural_results])
    df_results.to_csv(output_dir / 'before_after_metrics.csv', index=False)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    
    return all_results


if __name__ == '__main__':
    results = main()
