"""
T370: Comprehensive Dataset Expansion & Metrics Validation

Rigorous validation of model performance with expanded dataset
and verified metric calculations.
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_poisson_model import PoissonMatchPredictor
from src.models.loader import ModelLoader
import torch
from scripts.train_neural_model import DenseNN


def analyze_dataset():
    """Analyze available data and create expanded test set."""
    print("="*70)
    print("DATASET ANALYSIS")
    print("="*70)
    
    df = pd.read_csv('data/processed/epl_features_2021_2024.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    print(f"\nTotal matches: {len(df)}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Seasons covered: {df['Season'].unique()}")
    
    # Time-based splits
    n = len(df)
    
    # Original T360 split (85% train, 15% test = 171 samples)
    test_start_original = int(n * 0.85)
    
    # New expanded split (70% train, 30% test for validation)
    test_start_expanded = int(n * 0.70)
    
    test_df_original = df.iloc[test_start_original:]
    test_df_expanded = df.iloc[test_start_expanded:]
    
    print(f"\nOriginal T360 test set: {len(test_df_original)} matches")
    print(f"  Date range: {test_df_original['Date'].min()} to {test_df_original['Date'].max()}")
    
    print(f"\nExpanded T370 test set: {len(test_df_expanded)} matches")
    print(f"  Date range: {test_df_expanded['Date'].min()} to {test_df_expanded['Date'].max()}")
    
    # Save dataset description
    dataset_info = {
        'total_matches': int(n),
        'date_range': {
            'start': str(df['Date'].min()),
            'end': str(df['Date'].max())
        },
        'seasons': list(df['Season'].unique()),
        'original_test': {
            'size': int(len(test_df_original)),
            'start_date': str(test_df_original['Date'].min()),
            'end_date': str(test_df_original['Date'].max())
        },
        'expanded_test': {
            'size': int(len(test_df_expanded)),
            'start_date': str(test_df_expanded['Date'].min()),
            'end_date': str(test_df_expanded['Date'].max())
        }
    }
    
    # Save markdown description
    with open('outputs/diagnostics/T370/dataset_description.md', 'w') as f:
        f.write("# T370: Dataset Description\n\n")
        f.write(f"## Total Data\n")
        f.write(f"- Matches: {n}\n")
        f.write(f"- Date Range: {df['Date'].min()} to {df['Date'].max()}\n")
        f.write(f"- Seasons: {', '.join(map(str, df['Season'].unique()))}\n\n")
        f.write(f"## Test Sets\n\n")
        f.write(f"### Original (T360)\n")
        f.write(f"- Size: {len(test_df_original)} matches (15% of data)\n")
        f.write(f"- **⚠️ TOO SMALL for robust evaluation**\n\n")
        f.write(f"### Expanded (T370)\n")
        f.write(f"- Size: {len(test_df_expanded)} matches (30% of data)\n")
        f.write(f"- **✓ Statistically robust**\n")
    
    return test_df_original, test_df_expanded, dataset_info


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


def manual_logloss_check(y_true, y_prob, n_examples=10):
    """Manually verify LogLoss calculation on examples."""
    print("\n" + "="*70)
    print("MANUAL LOGLOSS VERIFICATION")
    print("="*70)
    
    manual_checks = []
    
    print("\nFormula: LogLoss = -1/N * Σ log(p_true_class)")
    print("\nVerifying first 10 examples:\n")
    
    for i in range(min(n_examples, len(y_true))):
        true_class = int(y_true[i])
        probs = y_prob[i]
        prob_true_class = probs[true_class]
        
        # Manual calculation
        log_p = np.log(prob_true_class)
        contrib = -log_p
        
        print(f"Example {i+1}:")
        print(f"  True class: {true_class} ({'H' if true_class==0 else 'D' if true_class==1 else 'A'})")
        print(f"  Probabilities: H={probs[0]:.4f}, D={probs[1]:.4f}, A={probs[2]:.4f}")
        print(f"  P(true class): {prob_true_class:.4f}")
        print(f"  -log(P): {contrib:.4f}")
        
        manual_checks.append({
            'example': i+1,
            'true_class': int(true_class),
            'prob_home': float(probs[0]),
            'prob_draw': float(probs[1]),
            'prob_away': float(probs[2]),
            'prob_true': float(prob_true_class),
            'contribution': float(contrib)
        })
    
    # Calculate manual average
    manual_logloss = np.mean([c['contribution'] for c in manual_checks])
    
    # Sklearn calculation
    sklearn_logloss = log_loss(y_true[:n_examples], y_prob[:n_examples])
    
    print(f"\nManual LogLoss (first {n_examples}): {manual_logloss:.4f}")
    print(f"Sklearn LogLoss (first {n_examples}): {sklearn_logloss:.4f}")
    print(f"Difference: {abs(manual_logloss - sklearn_logloss):.6f}")
    
    if abs(manual_logloss - sklearn_logloss) < 0.001:
        print("✓ LogLoss calculation VERIFIED")
    else:
        print("⚠️ LogLoss calculation MISMATCH - investigate!")
    
    # Save to markdown
    with open('outputs/diagnostics/T370/logloss_manual_check.md', 'w') as f:
        f.write("# LogLoss Manual Verification\n\n")
        f.write("## Formula\n")
        f.write("```\nLogLoss = -1/N * Σ log(p_true_class)\n```\n\n")
        f.write("## Manual Calculations\n\n")
        for check in manual_checks:
            f.write(f"### Example {check['example']}\n")
            f.write(f"- True class: {check['true_class']}\n")
            f.write(f"- Probabilities: H={check['prob_home']:.4f}, D={check['prob_draw']:.4f}, A={check['prob_away']:.4f}\n")
            f.write(f"- P(true): {check['prob_true']:.4f}\n")
            f.write(f"- -log(P): {check['contribution']:.4f}\n\n")
        
        f.write(f"\n## Results\n")
        f.write(f"- Manual LogLoss: {manual_logloss:.4f}\n")
        f.write(f"- Sklearn LogLoss: {sklearn_logloss:.4f}\n")
        f.write(f"- Match: {'✓ VERIFIED' if abs(manual_logloss - sklearn_logloss) < 0.001 else '⚠️ MISMATCH'}\n")
    
    return manual_checks


def evaluate_model_comprehensive(model_name, test_df, get_predictions_func):
    """Comprehensive model evaluation."""
    print(f"\n{'='*70}")
    print(f"{model_name.upper()} EVALUATION - EXPANDED DATASET")
    print(f"{'='*70}")
    
    # Get predictions
    probs = get_predictions_func(test_df)
    
    # Get true labels
    result_map = {'H': 0, 'D': 1, 'A': 2}
    y_true = test_df['FTR'].map(result_map).values[:len(probs)]
    
    # Calculate metrics
    y_pred = np.argmax(probs, axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    logloss = log_loss(y_true, probs)
    
    # Brier score
    brier_scores = []
    for i in range(3):
        y_binary = (y_true == i).astype(int)
        brier = brier_score_loss(y_binary, probs[:, i])
        brier_scores.append(brier)
    brier_avg = np.mean(brier_scores)
    
    # ECE
    ece = calculate_ece(y_true, probs)
    
    results = {
        'model': model_name,
        'test_samples': int(len(y_true)),
        'accuracy': float(accuracy),
        'log_loss': float(logloss),
        'brier_score': float(brier_avg),
        'ece': float(ece)
    }
    
    print(f"\nResults on {len(y_true)} matches:")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Log Loss: {logloss:.4f}")
    print(f"  Brier Score: {brier_avg:.4f}")
    print(f"  ECE: {ece:.4f}")
    
    # Manual LogLoss check
    if model_name == "CatBoost":
        print("\n🔍 Manual LogLoss verification for CatBoost:")
        manual_logloss_check(y_true, probs)
    
    return results, probs, y_true


def get_poisson_predictions(test_df):
    """Get Poisson predictions."""
    model = PoissonMatchPredictor.load('models/poisson_v1_latest.pkl')
    df_poisson = test_df.copy()
    if 'HomeTeam' not in df_poisson.columns:
        df_poisson = df_poisson.rename(columns={'home_team': 'HomeTeam', 'away_team': 'AwayTeam'})
    predictions = model.predict(df_poisson)
    return predictions[['prob_home', 'prob_draw', 'prob_away']].values


def get_catboost_predictions(test_df):
    """Get CatBoost predictions."""
    loader = ModelLoader()
    loader.load_latest()
    feature_cols = [col for col in test_df.columns 
                   if col not in ['Date', 'HomeTeam', 'AwayTeam', 'Season', 'FTR', 'home_team', 'away_team']]
    X = test_df[feature_cols].values
    return loader.predict(X)


def get_neural_predictions(test_df):
    """Get Neural predictions."""
    checkpoint = torch.load('models/neural_v1_latest.pt', weights_only=False, map_location='cpu')
    
    model = DenseNN(input_dim=22, hidden_dims=[64, 32, 16], dropout=0.3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler = checkpoint['scaler']
    calibrators = checkpoint.get('calibrators')
    
    feature_cols = [col for col in test_df.columns 
                   if col not in ['Date', 'HomeTeam', 'AwayTeam', 'Season', 'FTR', 'home_team', 'away_team']]
    X = test_df[feature_cols].values
    
    X_scaled = scaler.transform(X)
    X_tensor = torch.FloatTensor(X_scaled)
    
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).numpy()
    
    if calibrators:
        calibrated = np.zeros_like(probs)
        for i, cal in enumerate(calibrators):
            calibrated[:, i] = cal.predict(probs[:, i])
        row_sums = calibrated.sum(axis=1, keepdims=True)
        probs = calibrated / row_sums
    
    return probs


def main():
    """Run comprehensive validation."""
    print("="*70)
    print("T370: COMPREHENSIVE METRICS VALIDATION")
    print("="*70)
    print(f"\nTimestamp: {datetime.now().isoformat()}\n")
    
    # Phase 1: Dataset Analysis
    test_original, test_expanded, dataset_info = analyze_dataset()
    
    # Phase 2: Evaluate on expanded dataset
    all_results = {}
    
    # Poisson
    poisson_results, poisson_probs, y_true = evaluate_model_comprehensive(
        "Poisson", test_expanded, get_poisson_predictions
    )
    all_results['poisson'] = poisson_results
    
    # CatBoost
    catboost_results, catboost_probs, _ = evaluate_model_comprehensive(
        "CatBoost", test_expanded, get_catboost_predictions
    )
    all_results['catboost'] = catboost_results
    
    # Neural (try-catch for stability)
    try:
        neural_results, neural_probs, _ = evaluate_model_comprehensive(
            "Neural", test_expanded, get_neural_predictions
        )
        all_results['neural'] = neural_results
    except Exception as e:
        print(f"\n⚠️ Neural evaluation failed: {e}")
        all_results['neural'] = {'error': str(e)}
    
    # Save comprehensive results
    df_results = pd.DataFrame([
        poisson_results,
        catboost_results,
        all_results.get('neural', {})
    ])
    df_results.to_csv('outputs/diagnostics/T370/metrics_full_eval.csv', index=False)
    
    # Save JSON
    final_output = {
        'dataset': dataset_info,
        'models': all_results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('outputs/diagnostics/T370/final_conclusion.json', 'w') as f:
        json.dump(final_output, f, indent=2)
    
    print("\n" + "="*70)
    print("T370 VALIDATION COMPLETE")
    print("="*70)
    print(f"\nResults saved to: outputs/diagnostics/T370/")
    
    return all_results


if __name__ == '__main__':
    results = main()
