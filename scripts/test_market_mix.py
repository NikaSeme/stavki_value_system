
import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
import logging
from scipy.stats import poisson

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_catboost(models_dir):
    path = models_dir / 'catboost_v1_latest.pkl'
    calib_path = models_dir / 'calibrator_v1_latest.pkl'
    logger.info(f"Loading CatBoost: {path}")
    model = joblib.load(path)
    
    calibrator = None
    if calib_path.exists():
        logger.info(f"Loading Calibrator: {calib_path}")
        calibrator = joblib.load(calib_path)
    else:
        logger.warning("No calibrator found! Using raw probabilities.")
        
    return model, calibrator

def load_poisson(models_dir):
    path = models_dir / 'poisson_v1_latest.pkl'
    logger.info(f"Loading Poisson: {path}")
    with open(path, 'rb') as f:
        params = joblib.load(f)
    return params

def get_catboost_probs(model, calibrator, df, feature_names):
    # Ensure cols exist, fill 0 if missing
    X = df[feature_names].copy()
    for c in feature_names:
        if c not in X.columns: X[c] = 0
    
    # Use calibrator if available, otherwise raw model
    # Note: calibrator expects the same input as model or sometimes probas depending on implementation
    # Based on train_model.py, calibrator wraps the model? No, train_model says:
    # calibrator.fit(X_val, y_val)
    # evaluate_model(calibrator, X_test, ...)
    # So calibrator expects X input!
    
    if calibrator:
        return calibrator.predict_proba(X)
    else:
        return model.predict_proba(X)

def get_poisson_probs(params, df):
    home_adv = params['home_advantage']
    att = params['team_attack']
    defe = params['team_defense']
    avg_goals = params['league_avg_goals']
    
    probs = []
    max_goals = 6
    
    for _, row in df.iterrows():
        ht, at = row['HomeTeam'], row['AwayTeam']
        h_att = att.get(ht, 1.0)
        h_def = defe.get(ht, 1.0)
        a_att = att.get(at, 1.0)
        a_def = defe.get(at, 1.0)
        
        lam_h = avg_goals * h_att * a_def * (1 + home_adv)
        lam_a = avg_goals * a_att * h_def
        
        p_h, p_d, p_a = 0, 0, 0
        for i in range(max_goals):
            for j in range(max_goals):
                p = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
                if i > j: p_h += p
                elif i == j: p_d += p
                else: p_a += p
        probs.append([p_h, p_d, p_a])
        
    return np.array(probs)

def calculate_implied_probs(odds):
    # odds shape: (N, 3) -> H, D, A
    # implied prob = 1 / odds
    probs = 1.0 / odds
    # Normalize to remove margin
    row_sums = probs.sum(axis=1, keepdims=True)
    probs_norm = probs / row_sums
    return probs_norm

def main():
    base_dir = Path(__file__).parent.parent
    data_file = base_dir / 'data' / 'processed' / 'multi_league_features_2021_2024.csv'
    models_dir = base_dir / 'models'
    optimized_meta = models_dir / 'ensemble_optimized_metadata.json' # Created by optimize script
    
    if not optimized_meta.exists():
        logger.error("Optimization metadata not found! Run optimize_ensemble_weights.py first.")
        return

    # Load Weights
    with open(optimized_meta) as f:
        meta = json.load(f)
        weights = meta['weights']
        w_c = weights['catboost']
        w_p = weights['poisson']
        # Neural is 0.0 effectively
        logger.info(f"Loaded Optimal Model Weights: CatBoost={w_c:.2f}, Poisson={w_p:.2f}")

    # Load Data
    df = pd.read_csv(data_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Load Models
    catboost_model, calibrator = load_catboost(models_dir)
    poisson_params = load_poisson(models_dir)
    
    with open(models_dir / 'metadata_v1_latest.json') as f:
        cb_features = json.load(f)['features']

    # Generate Model Predictions
    # To save time, we calculate on Validation set + Test set only (last 30%)
    n = len(df)
    test_start = int(n * 0.70)
    df_test = df.iloc[test_start:].copy()
    
    logger.info(f"Testing Market Mix on last {len(df_test)} matches...")
    
    p_cb = get_catboost_probs(catboost_model, calibrator, df_test, cb_features)
    p_ps = get_poisson_probs(poisson_params, df_test)
    
    # Ensemble Probability
    p_ensemble = w_c * p_cb + w_p * p_ps
    row_sums = p_ensemble.sum(axis=1, keepdims=True)
    p_ensemble /= row_sums
    
    # Market Probability (Implied)
    odds_cols = ['OddsHome', 'OddsDraw', 'OddsAway']
    odds = df_test[odds_cols].values
    p_market = calculate_implied_probs(odds)
    
    # Target
    result_map = {'H': 0, 'D': 1, 'A': 2}
    y_true = df_test['FTR'].map(result_map).values
    
    # Test Alphas
    alphas = np.linspace(0.0, 1.0, 11) # 0.0, 0.1, ... 1.0
    
    results = []
    
    logger.info("\nTesting Alpha Blending (Model vs Market)...")
    logger.info(f"{'Alpha (Model%)':<15} {'Log Loss':<10} {'ROI% (Ev>5%)':<15} {'Bets':<10}")
    logger.info("-" * 50)
    
    for alpha in alphas:
        # P_final = alpha * Model + (1-alpha) * Market
        p_final = alpha * p_ensemble + (1 - alpha) * p_market
        row_sums = p_final.sum(axis=1, keepdims=True)
        p_final /= row_sums # Renormalize just in case
        
        # Log Loss
        try:
            from sklearn.metrics import log_loss
            ll = log_loss(y_true, p_final)
        except:
            ll = 9.99
            
        # ROI Calculation
        # Bet if EV > X%
        ev = (p_final * odds) - 1
        bet_mask = ev > 0.05
        
        profit = 0.0
        staked = 0.0
        bets_count = 0
        
        # Vectorized ROI
        for i in range(3):
            mask = bet_mask[:, i]
            n_bets = mask.sum()
            if n_bets == 0: continue
            
            won = (y_true == i)
            pnl = np.where(won[mask], odds[mask, i] - 1, -1.0)
            profit += pnl.sum()
            staked += n_bets
            bets_count += n_bets
            
        roi = (profit / staked * 100) if staked > 0 else 0.0
        
        logger.info(f"{alpha:<15.1f} {ll:<10.4f} {roi:<15.2f} {bets_count:<10}")
        results.append({'alpha': alpha, 'roi': roi, 'log_loss': ll})
        
    best_roi = max(results, key=lambda x: x['roi'])
    best_ll = min(results, key=lambda x: x['log_loss'])
    
    logger.info("\nSummary:")
    logger.info(f"Best ROI Alpha: {best_roi['alpha']:.1f} (ROI: {best_roi['roi']:.2f}%)")
    logger.info(f"Best LogLoss Alpha: {best_ll['alpha']:.1f} (LogLoss: {best_ll['log_loss']:.4f})")
    
    # Recommendation
    if best_roi['alpha'] == 1.0:
        logger.info("Recommendation: Ignore Market! Pure Value Betting.")
    elif best_roi['alpha'] == 0.0:
        logger.info("Recommendation: Follow Market! (Or adjust threshold)")
    else:
        logger.info(f"Recommendation: Mix {best_roi['alpha']*100:.0f}% Model + {(1-best_roi['alpha'])*100:.0f}% Market")

if __name__ == "__main__":
    main()
