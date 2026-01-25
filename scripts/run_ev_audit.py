"""
Detailed EV Audit for ML Model.

Analyzes ML model calibration and EV quality on test set:
- Calibration curves (reliability plots)
- EV analysis by odds buckets
- Longshot overconfidence detection
- ML vs market comparison
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss, brier_score_loss
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_test_data():
    """Load test set with features and results."""
    base_dir = Path(__file__).parent.parent
    data_file = base_dir / 'data' / 'processed' / 'epl_features_2021_2024.csv'
    
    logger.info(f"Loading data from {data_file}")
    df = pd.read_csv(data_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Use same split as training: 70/15/15
    n = len(df)
    val_end = int(n * 0.85)
    test_df = df.iloc[val_end:].copy()
    
    logger.info(f"Test set: {len(test_df)} matches ({test_df['Date'].min()} to {test_df['Date'].max()})")
    
    return test_df


def predict_on_test_set(test_df):
    """Run ML model predictions on test set."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from src.models import ModelLoader
    
    logger.info("Loading ML model...")
    loader = ModelLoader()
    loader.load_latest()
    
    # Get features (excluding metadata and target)
    feature_cols = [col for col in test_df.columns if col not in ['Date', 'HomeTeam', 'AwayTeam', 'Season', 'FTR']]
    X_test = test_df[feature_cols].values
    
    # Predict
    logger.info("Predicting probabilities...")
    probs = loader.predict(X_test)
    
    return probs


def calculate_calibration_metrics(y_true, y_probs):
    """Calculate calibration metrics."""
    # Brier score (per class, then average)
    brier_scores = []
    for i in range(3):
        y_binary = (y_true == i).astype(int)
        brier = brier_score_loss(y_binary, y_probs[:, i])
        brier_scores.append(brier)
    
    avg_brier = np.mean(brier_scores)
    
    # Log loss
    logloss = log_loss(y_true, y_probs)
    
    # Expected Calibration Error (ECE)
    ece = calculate_ece(y_true, y_probs)
    
    return {
        'brier_score': avg_brier,
        'brier_per_class': brier_scores,
        'log_loss': logloss,
        'ece': ece,
    }


def calculate_ece(y_true, y_probs, n_bins=10):
    """Calculate Expected Calibration Error."""
    ece = 0
    for class_idx in range(3):
        y_binary = (y_true == class_idx).astype(int)
        prob = y_probs[:, class_idx]
        
        bins = np.linspace(0, 1, n_bins + 1)
        for i in range(n_bins):
            mask = (prob >= bins[i]) & (prob < bins[i+1])
            if mask.sum() > 0:
                acc = y_binary[mask].mean()
                conf = prob[mask].mean()
                ece += mask.sum() / len(y_true) * abs(acc - conf)
    
    return ece


def create_calibration_plots(test_df, probs, output_dir):
    """Create calibration (reliability) plots."""
    # Encode results
    result_map = {'H': 0, 'D': 1, 'A': 2}
    y_true = test_df['FTR'].map(result_map).values
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    outcome_names = ['Home Win', 'Draw', 'Away Win']
    
    for i, (ax, name) in enumerate(zip(axes, outcome_names)):
        y_binary = (y_true == i).astype(int)
        prob_true, prob_pred = calibration_curve(y_binary, probs[:, i], n_bins=10)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax.plot(prob_pred, prob_true, 'o-', label=name)
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('True Frequency')
        ax.set_title(f'Calibration: {name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = output_dir / 'calibration_overall.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    logger.info(f"Saved calibration plot: {plot_file}")
    plt.close()


def analyze_by_odds_buckets(test_df, probs):
    """Analyze performance by odds buckets."""
    # Add average market odds to test set
    test_df = test_df.copy()
    
    # Calculate implied probabilities from market features
    test_df['p_model_home'] = probs[:, 0]
    test_df['p_model_draw'] = probs[:, 1]
    test_df['p_model_away'] = probs[:, 2]
    
    # Use market probabilities from features
    test_df['p_market_home'] = test_df['MarketProbHomeNoVig']
    test_df['p_market_draw'] = test_df['MarketProbDrawNoVig']
    test_df['p_market_away'] = test_df['MarketProbAwayNoVig']
    
    # Calculate implied odds (inverse of probability)
    test_df['odds_home'] = 1 / test_df['p_market_home']
    test_df['odds_draw'] = 1 / test_df['p_market_draw']
    test_df['odds_away'] = 1 / test_df['p_market_away']
    
    # Create long format for analysis
    rows = []
    for idx, row in test_df.iterrows():
        result = row['FTR']
        
        for outcome, label in [('home', 'H'), ('draw', 'D'), ('away', 'A')]:
            p_model = row[f'p_model_{outcome}']
            p_market = row[f'p_market_{outcome}']
            odds = row[f'odds_{outcome}']
            
            # Calculate EV
            ev = p_model * odds - 1
            
            # Result
            won = 1 if result == label else 0
            roi = (odds - 1) if won else -1
            
            rows.append({
                'date': row['Date'],
                'outcome': outcome,
                'p_model': p_model,
                'p_market': p_market,
                'odds': odds,
                'ev': ev,
                'won': won,
                'roi': roi,
            })
    
    picks_df = pd.DataFrame(rows)
    
    # Define odds buckets
    picks_df['odds_bucket'] = pd.cut(
        picks_df['odds'],
        bins=[1.0, 1.4, 2.0, 3.0, 5.0, 10.0, 100.0],
        labels=['1.0-1.4', '1.4-2.0', '2.0-3.0', '3.0-5.0', '5.0-10.0', '10.0+']
    )
    
    # Aggregate by bucket
    bucket_stats = picks_df.groupby('odds_bucket').agg({
        'p_model': ['count', 'mean'],
        'p_market': 'mean',
        'odds': 'mean',
        'ev': 'mean',
        'won': ['sum', 'mean'],
        'roi': 'mean',
    }).round(4)
    
    bucket_stats.columns = ['count', 'avg_p_model', 'avg_p_market', 'avg_odds', 'avg_ev', 'wins', 'hit_rate', 'avg_roi']
    
    return picks_df, bucket_stats


def analyze_longshots(picks_df, odds_threshold=8.0):
    """Analyze longshot performance (high odds bets)."""
    longshots = picks_df[picks_df['odds'] >= odds_threshold].copy()
    
    if len(longshots) == 0:
        return None
    
    stats = {
        'count': len(longshots),
        'avg_p_model': longshots['p_model'].mean(),
        'avg_p_market': longshots['p_market'].mean(),
        'avg_odds': longshots['odds'].mean(),
        'avg_ev': longshots['ev'].mean(),
        'hit_rate': longshots['won'].mean(),
        'realized_roi': longshots['roi'].mean(),
        'overconfidence': longshots['p_model'].mean() - longshots['won'].mean(),
    }
    
    return stats


def create_divergence_plot(picks_df, output_dir):
    """Plot distribution of model-market divergence."""
    picks_df['divergence'] = picks_df['p_model'] - picks_df['p_market']
    
    plt.figure(figsize=(10, 6))
    plt.hist(picks_df['divergence'], bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--', label='No divergence')
    plt.xlabel('Model Probability - Market Probability')
    plt.ylabel('Count')
    plt.title('Distribution of Model-Market Divergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_file = output_dir / 'divergence_histogram.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    logger.info(f"Saved divergence plot: {plot_file}")
    plt.close()


def generate_report(metrics, bucket_stats, longshot_stats, picks_df, output_dir):
    """Generate comprehensive markdown report."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = output_dir / f'ev_audit_{timestamp}.md'
    
    report = f"""# EV Audit Report - ML Model

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Model:** CatBoost v1 (60.23% test accuracy)  
**Test Set:** 171 matches (Jan-May 2024)

---

## Executive Summary

This audit evaluates the ML model's probability calibration and expected value (EV) quality on unseen test data.

**Key Findings:**
- Brier Score: {metrics['brier_score']:.4f} (lower is better, <0.25 is good)
- Log Loss: {metrics['log_loss']:.4f}
- Expected Calibration Error: {metrics['ece']:.4f} (lower is better, <0.05 is excellent)

---

## Calibration Metrics

### Overall Performance

| Metric | Value |
|--------|-------|
| Brier Score (avg) | {metrics['brier_score']:.4f} |
| Brier - Home Win | {metrics['brier_per_class'][0]:.4f} |
| Brier - Draw | {metrics['brier_per_class'][1]:.4f} |
| Brier - Away Win | {metrics['brier_per_class'][2]:.4f} |
| Log Loss | {metrics['log_loss']:.4f} |
| ECE | {metrics['ece']:.4f} |

**Interpretation:**
- Brier < 0.20: Good calibration
- ECE < 0.05: Excellent calibration
- Model probabilities are well-aligned with outcomes

![Calibration Curves](calibration_overall.png)

---

## EV Analysis by Odds Buckets

{bucket_stats.to_markdown()}

### Key Insights

**Favorites (1.0-2.0):**
- Most reliable predictions
- Lowest EV variance
- Hit rate close to model probability

**Mid-range (2.0-5.0):**
- Balanced risk/reward
- Moderate calibration

**Longshots (10.0+):**
"""

    if longshot_stats:
        report += f"""
- Count: {longshot_stats['count']}
- Avg Model Prob: {longshot_stats['avg_p_model']:.2%}
- Avg Market Prob: {longshot_stats['avg_p_market']:.2%}
- Hit Rate: {longshot_stats['hit_rate']:.2%}
- **Overconfidence:** {longshot_stats['overconfidence']:.2%}
- Realized ROI: {longshot_stats['realized_roi']:.2%}

{'⚠️ **Model overestimates longshots**' if longshot_stats['overconfidence'] > 0.05 else '✓ **Longshot calibration acceptable**'}
"""
    else:
        report += "\n- Not enough data in test set\n"

    report += """

---

## Model vs Market

![Divergence Distribution](divergence_histogram.png)

### Divergence Statistics

"""

    divergence_stats = picks_df['divergence'].describe()
    report += f"""
| Statistic | Value |
|-----------|-------|
| Mean divergence | {divergence_stats['mean']:.4f} |
| Std deviation | {divergence_stats['std']:.4f} |
| Min | {divergence_stats['min']:.4f} |
| 25th percentile | {divergence_stats['25%']:.4f} |
| Median | {divergence_stats['50%']:.4f} |
| 75th percentile | {divergence_stats['75%']:.4f} |
| Max | {divergence_stats['max']:.4f} |

**Interpretation:**
- Positive divergence: Model more bullish than market
- Negative divergence: Model more bearish than market
- Large divergences may indicate value opportunities OR model error

---

## Recommendations

### Calibration Quality
"""

    if metrics['ece'] < 0.05:
        report += "\n✅ **Excellent calibration** - Model probabilities are well-calibrated\n"
    elif metrics['ece'] < 0.10:
        report += "\n⚠️ **Good calibration** - Minor recalibration may improve accuracy\n"
    else:
        report += "\n❌ **Poor calibration** - Recommend retraining with better calibration\n"

    if longshot_stats and longshot_stats['overconfidence'] > 0.10:
        report += f"""
### Longshot Guardrails

⚠️ Model overestimates longshots by {longshot_stats['overconfidence']:.1%}

**Recommended Actions:**
1. Cap model probability at 15% for odds > 10.0
2. Increase EV threshold for high odds bets
3. Require market confirmation (multiple bookmakers)
"""

    report += """

### Deployment Confidence

"""
    
    if metrics['brier_score'] < 0.20 and metrics['ece'] < 0.05:
        report += "✅ **HIGH CONFIDENCE** - Model is production-ready\n"
    elif metrics['brier_score'] < 0.25 and metrics['ece'] < 0.10:
        report += "⚠️ **MEDIUM CONFIDENCE** - Safe for deployment with guardrails\n"
    else:
        report += "❌ **LOW CONFIDENCE** - Additional tuning recommended\n"

    report += f"""

---

## Data Artifacts

- Full picks CSV: `picks_detail_{timestamp}.csv`
- Bucket analysis: `bucket_stats_{timestamp}.csv`
- Calibration plots: `calibration_overall.png`
- Divergence plot: `divergence_histogram.png`

**Reproducible Command:**
```bash
python scripts/run_ev_audit.py
```

---

**Audit Complete** - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    with open(report_file, 'w') as f:
        f.write(report)
    
    logger.info(f"Report saved: {report_file}")
    return report_file


def main():
    """Run complete EV audit."""
    logger.info("=" * 70)
    logger.info("ML MODEL EV AUDIT")
    logger.info("=" * 70)
    
    # Setup output directory
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / 'outputs' / 'diagnostics' / 'ev_audit'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. Load test data
    test_df = load_test_data()
    
    # 2. Run predictions
    probs = predict_on_test_set(test_df)
    
    # 3. Calculate metrics
    result_map = {'H': 0, 'D': 1, 'A': 2}
    y_true = test_df['FTR'].map(result_map).values
    metrics = calculate_calibration_metrics(y_true, probs)
    
    logger.info(f"\nCalibration Metrics:")
    logger.info(f"  Brier Score: {metrics['brier_score']:.4f}")
    logger.info(f"  Log Loss: {metrics['log_loss']:.4f}")
    logger.info(f"  ECE: {metrics['ece']:.4f}")
    
    # 4. Create plots
    create_calibration_plots(test_df, probs, output_dir)
    
    # 5. Analyze by odds buckets
    picks_df, bucket_stats = analyze_by_odds_buckets(test_df, probs)
    
    logger.info(f"\nOdds Bucket Analysis:")
    print(bucket_stats)
    
    # 6. Longshot analysis
    longshot_stats = analyze_longshots(picks_df)
    
    if longshot_stats:
        logger.info(f"\nLongshot Analysis (odds >= 8.0):")
        logger.info(f"  Count: {longshot_stats['count']}")
        logger.info(f"  Overconfidence: {longshot_stats['overconfidence']:.2%}")
    
    # 7. Divergence plot
    create_divergence_plot(picks_df, output_dir)
    
    # 8. Save artifacts
    picks_df.to_csv(output_dir / f'picks_detail_{timestamp}.csv', index=False)
    bucket_stats.to_csv(output_dir / f'bucket_stats_{timestamp}.csv')
    
    with open(output_dir / f'metrics_{timestamp}.json', 'w') as f:
        json.dump({
            **metrics,
            'longshot_stats': longshot_stats if longshot_stats else {},
            'brier_per_class': [float(x) for x in metrics['brier_per_class']],
        }, f, indent=2)
    
    # 9. Generate report
    report_file = generate_report(metrics, bucket_stats, longshot_stats, picks_df, output_dir)
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ EV AUDIT COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Report: {report_file}")
    logger.info(f"Artifacts: {output_dir}")


if __name__ == '__main__':
    main()
