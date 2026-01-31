
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from itertools import product
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.strategy.blending import get_liquidity_tier

class MetaOptimizer:
    """
    Adaptive Meta-Learning Optimizer.
    Tunes:
    1. External Alpha (Model vs Market weight)
    2. Internal Ensemble Weights (CatBoost vs Neural vs Poisson)
    
    Strategy:
    - Big Leagues (> MIN_MATCHES): Optimized individually.
    - Small Leagues: Optimized by Liquidity Tier group.
    """
    
    MIN_MATCHES = 500  # Threshold to be treated as "Big League"
    MAX_DRAWDOWN = -0.30 # Max allowed drawdown (30%)
    
    def __init__(self, history_path: str = "data/processed/multi_league_master_peopled.csv"):
        self.history_path = history_path
        self.df = None
        self.results = {}
        
    def load_data(self):
        """Load and preprocess 2021-2024 dataset."""
        if not os.path.exists(self.history_path):
            logger.error(f"History file not found: {self.history_path}")
            sys.exit(1)
            
        logger.info(f"Loading history from {self.history_path}...")
        self.df = pd.read_csv(self.history_path)
        
        # Renaissance mapping
        rename_map = {
            'HomeTeam': 'home_team', 'AwayTeam': 'away_team', 'Date': 'commence_time',
            'League': 'sport_key' # Ensure League column exists in CSV or map it
        }
        self.df = self.df.rename(columns=rename_map)
        
        # Synthetic League/Sport Key if missing
        if 'sport_key' not in self.df.columns:
            # Try to infer or fail
            # For multi_league_features_6leagues_full.csv, 'League' usually exists
            logger.warning("'sport_key' column missing. Attempting to use 'League'...")
            if 'League' in self.df.columns:
                 self.df['sport_key'] = self.df['League']
            else:
                 logger.error("Dataset missing 'League' or 'sport_key'. Cannot optimize per league.")
                 sys.exit(1)
        
        # OMNI-PATCH: Fill ValueOdds from alternative columns if AvgOdds missing
        # Championship/Ligue1/SerieA often have 'OddsHome' but not 'AvgOddsH'
        if 'OddsHome' in self.df.columns:
            # 1. Fill AvgOdds from OddsHome (Fixes Champ/Ligue1)
            self.df['AvgOddsH'] = self.df['AvgOddsH'].fillna(self.df['OddsHome'])
            self.df['AvgOddsD'] = self.df['AvgOddsD'].fillna(self.df['OddsDraw'])
            self.df['AvgOddsA'] = self.df['AvgOddsA'].fillna(self.df['OddsAway'])
            
            # 2. Fill OddsHome from AvgOdds (Fixes EPL)
            # Neural likely expects OddsHome if it was in training set
            self.df['OddsHome'] = self.df['OddsHome'].fillna(self.df['AvgOddsH'])
            self.df['OddsDraw'] = self.df['OddsDraw'].fillna(self.df['AvgOddsD'])
            self.df['OddsAway'] = self.df['OddsAway'].fillna(self.df['AvgOddsA'])
        
        # CLEAN DATA: Drop rows with missing Market Odds
        initial_len = len(self.df)
        self.df = self.df.dropna(subset=['AvgOddsH', 'AvgOddsD', 'AvgOddsA'])
        cleaned_len = len(self.df)
        
        if cleaned_len < initial_len:
            logger.warning(f"Dropped {initial_len - cleaned_len} rows due to missing AvgOdds. Remaining: {cleaned_len}")
            
        if cleaned_len == 0:
            logger.error("No valid data remaining after cleaning!")
            sys.exit(1)
                 
        logger.info(f"Loaded {len(self.df)} clean matches.")

        # Prepare Probabilities
        self._generate_model_probs()
        self._prepare_market_probs()
        
    def _generate_model_probs(self):
        """Generate raw model predictions using Augmented Historical Features."""
        logger.info("Generating Model Predictions (Ensemble)...")
        from src.models.ensemble_predictor import EnsemblePredictor
        ensemble = EnsemblePredictor()
        
        # Prepare Metadata
        meta = self.df[['home_team', 'away_team', 'commence_time']].copy()
        
        # Add synthetic event_id (LiveFeatureExtractor needs it)
        if 'event_id' not in meta.columns:
            meta['event_id'] = [f"hist_{i}" for i in range(len(meta))]
            # Also add to self.df for later use in feature extraction/alignment
            self.df['event_id'] = meta['event_id'].values
        
        # AUGMENT FEATURES:
        # The CSV has Elo/Form (Good), but lacks Market Features (Bad).
        # We must calculate Market Features and append them to self.df
        # so we can pass the FULL dataframe as 'features' to the ensemble.
        
        # 1. Market Features Calculation (Vectorized)
        def safe_div(x): return 1/x if x > 1 else 0
        
        # Basic Probs
        p_h = self.df['AvgOddsH'].apply(safe_div)
        p_d = self.df['AvgOddsD'].apply(safe_div)
        p_a = self.df['AvgOddsA'].apply(safe_div)
        
        # Normalize (Remove Vig)
        sums = p_h + p_d + p_a
        sums[sums == 0] = 1.0
        
        self.df['MarketProbHomeNoVig'] = p_h / sums
        self.df['MarketProbDrawNoVig'] = p_d / sums
        self.df['MarketProbAwayNoVig'] = p_a / sums
        
        # Odds Ratio
        # Avoid division by zero
        odds_a_safe = self.df['AvgOddsA'].replace(0, 1.0)
        self.df['OddsHomeAwayRatio'] = self.df['AvgOddsH'] / odds_a_safe
        
        # 2. Line Movement Features (Defaults)
        # Neural might expect these if trained on them
        if 'sharp_move_detected' not in self.df.columns: self.df['sharp_move_detected'] = 0
        if 'odds_volatility' not in self.df.columns: self.df['odds_volatility'] = 0.0
        if 'time_to_match_hours' not in self.df.columns: self.df['time_to_match_hours'] = 24.0
        if 'market_efficiency_score' not in self.df.columns: self.df['market_efficiency_score'] = 0.95
        
        # 3. Ensure Categoricals for CatBoost
        if 'League' not in self.df.columns: self.df['League'] = self.df['sport_key']
        # Season is likely in CSV, else default
        if 'Season' not in self.df.columns: self.df['Season'] = '2023-24'

        # CRITICAL FIX: Define exact 22 features for Neural Network
        # The scaler is blind (no feature names), so we MUST pass exactly 22 columns in correct order.
        # Based on V1 training schema (Elo + Form + Market + Fatigue)
        NEURAL_FEATURES = [
            'HomeEloBefore', 'AwayEloBefore', 'EloDiff',
            'Home_Pts_L5', 'Home_GF_L5', 'Home_GA_L5',
            'Away_Pts_L5', 'Away_GF_L5', 'Away_GA_L5',
            'Home_Overall_Pts_L5', 'Home_Overall_GF_L5', 'Home_Overall_GA_L5',
            'Away_Overall_Pts_L5', 'Away_Overall_GF_L5', 'Away_Overall_GA_L5',
            'WinStreak_L5', 'LossStreak_L5',
            'DaysSinceLastMatch',
            'MarketProbHomeNoVig', 'MarketProbDrawNoVig', 'MarketProbAwayNoVig', 'OddsHomeAwayRatio'
        ]
        
        # Verify columns exist - No longer filling with 1500 as CSV is "Peopled"
        for col in NEURAL_FEATURES:
            if col not in self.df.columns:
                 logger.error(f"REQUIRED FEATURE MISSING: {col}")
                 self.df[col] = 0.0

        try:
            # Construct strictly clean DF for Neural Predictor
            # Must include metadata for CatBoost/Ensemble wrapper
            # 'League' is needed by Catboost. 'home_team' etc by framework.
            
            # Ensure event_id is in self.df (it was assigned to meta, maybe not self.df properly?)
            if 'event_id' not in self.df.columns:
                 self.df['event_id'] = meta['event_id'].values

            cols_to_keep = NEURAL_FEATURES + ['sport_key', 'League', 'Season', 'home_team', 'away_team', 'commence_time', 'event_id']
            
            clean_features_df = self.df[cols_to_keep].copy()
            
            probs, components = ensemble.predict(meta, None, features=clean_features_df)
            
            # Store Component Probs
            self.df['cb_h'] = components['catboost'][:, 0]
            self.df['cb_d'] = components['catboost'][:, 1]
            self.df['cb_a'] = components['catboost'][:, 2]
            
            self.df['ps_h'] = components['poisson'][:, 0]
            self.df['ps_d'] = components['poisson'][:, 1]
            self.df['ps_a'] = components['poisson'][:, 2]
            
            if 'neural' in components and components['neural'] is not None:
                self.df['nn_h'] = components['neural'][:, 0]
                self.df['nn_d'] = components['neural'][:, 1]
                self.df['nn_a'] = components['neural'][:, 2]
                self.has_neural = True
                logger.info("✓ Neural Predictions Generated Successfully")
            else:
                self.has_neural = False
                self.df['nn_h'] = 0; self.df['nn_d'] = 0; self.df['nn_a'] = 0
                logger.warning("Neural Predictions missing (Model not loaded or failed)")
                
        except Exception as e:
            logger.error(f"Prediction generation failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def _prepare_market_probs(self):
        """Calculate No-Vig Market Probabilities from AvgOdds."""
        def safe_div(x): return 1/x if x > 1 else 0
        
        self.df['mk_h'] = self.df['AvgOddsH'].apply(safe_div)
        self.df['mk_d'] = self.df['AvgOddsD'].apply(safe_div)
        self.df['mk_a'] = self.df['AvgOddsA'].apply(safe_div)
        
        # Normalize
        sums = self.df[['mk_h', 'mk_d', 'mk_a']].sum(axis=1)
        sums[sums == 0] = 1.0
        self.df['mk_h'] /= sums
        self.df['mk_d'] /= sums
        self.df['mk_a'] /= sums

    def optimize(self):
        """Run the optimization process."""
        gb = self.df.groupby('sport_key')
        league_stats = gb.size()
        
        big_leagues = league_stats[league_stats >= self.MIN_MATCHES].index.tolist()
        small_leagues = league_stats[league_stats < self.MIN_MATCHES].index.tolist()
        
        logger.info(f"Optimization Plan: {len(big_leagues)} Big Leagues, {len(small_leagues)} Small Leagues.")
        
        weights_config = {}
        
        # 1. Optimize Big Leagues
        for league in big_leagues:
            logger.info(f"Optimizing Big League: {league} ({league_stats[league]} matches)...")
            sub_df = self.df[self.df['sport_key'] == league].copy()
            best_params = self._grid_search(sub_df)
            weights_config[league] = best_params
            
        # 2. Optimize Small Leagues (Grouped by Tier)
        # Create groups
        tier_groups = {'tier1': [], 'tier2': [], 'tier3': []}
        for league in small_leagues:
            tier = get_liquidity_tier(league) # This needs proper implementation in blending.py
            # Warning: get_liquidity_tier currently might return hardcoded stuff.
            # Assuming it works for standard keys. 
            # If our CSV has weird keys (e.g. 'D1'), we map them manually or rely on fallback.
            # For simulation, let's treat unknown as tier3.
            tier_groups.get(tier, tier_groups['tier3']).append(league)
            
        for tier, leagues in tier_groups.items():
            if not leagues: continue
            logger.info(f"Optimizing {tier} Bunch ({len(leagues)} leagues)...")
            sub_df = self.df[self.df['sport_key'].isin(leagues)].copy()
            if len(sub_df) < 50:
                logger.warning(f"Not enough data for {tier} ({len(sub_df)} matches). Skipping.")
                continue
                
            best_params = self._grid_search(sub_df)
            
            # Apply to all leagues in this tier
            for l in leagues:
                weights_config[l] = best_params
                
        # Save Results
        self._save_config(weights_config)

    def _grid_search(self, df: pd.DataFrame) -> Dict:
        """Find best Alpha and Internal Weights using 2-Stage Smart Zoom (Simplex)."""
        
        # Internal Weights Grid (Coarse Candidates)
        # We start with these anchors to find the general "Region of Interest"
        if self.has_neural:
            internal_mixes = [
                (0.5, 0.3, 0.2), # Default
                (0.7, 0.2, 0.1), # Heavy CatBoost
                (0.3, 0.6, 0.1), # Heavy Neural
                (0.4, 0.4, 0.2), # Balanced ML
                (0.2, 0.6, 0.2), # Aggressive Neural
                (0.8, 0.1, 0.1), # Aggressive CatBoost
                (0.33, 0.33, 0.34) # Equal
            ]
        else:
             internal_mixes = [
                 (0.7, 0.0, 0.3), # Default 2-model
                 (0.5, 0.0, 0.5), # Balanced
                 (0.9, 0.0, 0.1), # Heavy CatBoost
                 (0.3, 0.0, 0.7)  # Heavy Poisson
             ]
             
        # Pre-calculate components for speed
        cb_h, cb_d, cb_a = df['cb_h'].values, df['cb_d'].values, df['cb_a'].values
        nn_h, nn_d, nn_a = df['nn_h'].values, df['nn_d'].values, df['nn_a'].values
        ps_h, ps_d, ps_a = df['ps_h'].values, df['ps_d'].values, df['ps_a'].values
        mk_h, mk_d, mk_a = df['mk_h'].values, df['mk_d'].values, df['mk_a'].values
        odds_h, odds_d, odds_a = df['AvgOddsH'].values, df['AvgOddsD'].values, df['AvgOddsA'].values
        
        # Encode results: 0=H, 1=D, 2=A
        y_true = np.zeros(len(df), dtype=int)
        y_true[df['FTR'] == 'H'] = 0
        y_true[df['FTR'] == 'D'] = 1
        y_true[df['FTR'] == 'A'] = 2
        
        def run_sweep(alphas_to_test, weights_to_test):
            best_local_pnl = -float('inf')
            best_local_params = None
            
            for alpha in alphas_to_test:
                # Alpha precision to 3 decimals (0.1%)
                alpha = round(alpha, 3)
                
                for (wc, wn, wp) in weights_to_test:
                    # Enforce precision on weights too
                    wc, wn, wp = round(wc, 3), round(wn, 3), round(wp, 3)
                    
                    # 1. Calc Internal Ensemble Prob (Vectorized)
                    p_ens_h = cb_h*wc + nn_h*wn + ps_h*wp
                    p_ens_d = cb_d*wc + nn_d*wn + ps_d*wp
                    p_ens_a = cb_a*wc + nn_a*wn + ps_a*wp
                    
                    # Normalize (Internal)
                    s = p_ens_h + p_ens_d + p_ens_a
                    s[s==0] = 1.0
                    p_ens_h /= s; p_ens_d /= s; p_ens_a /= s
                    
                    # 2. Calc Final Prob (External Alpha)
                    p_fin_h = alpha * p_ens_h + (1-alpha) * mk_h
                    p_fin_d = alpha * p_ens_d + (1-alpha) * mk_d
                    p_fin_a = alpha * p_ens_a + (1-alpha) * mk_a
                    
                    # 3. Simulate PnL (Vectorized)
                    pnl_total = 0.0
                    
                    # EV Calculation
                    # H
                    ev_h = (p_fin_h * odds_h) - 1
                    bet_h = ev_h > 0.05
                    pnl_total += np.sum(odds_h[bet_h & (y_true==0)] - 1)
                    pnl_total -= np.sum(bet_h & (y_true!=0))
                    
                    # D
                    ev_d = (p_fin_d * odds_d) - 1
                    bet_d = ev_d > 0.05
                    pnl_total += np.sum(odds_d[bet_d & (y_true==1)] - 1)
                    pnl_total -= np.sum(bet_d & (y_true!=1))

                    # A
                    ev_a = (p_fin_a * odds_a) - 1
                    bet_a = ev_a > 0.05
                    pnl_total += np.sum(odds_a[bet_a & (y_true==2)] - 1)
                    pnl_total -= np.sum(bet_a & (y_true!=2))
                    
                    if pnl_total > best_local_pnl:
                        best_local_pnl = pnl_total
                        best_local_params = {
                            'alpha': alpha,
                            'weights': {'catboost': wc, 'neural': wn, 'poisson': wp},
                            'pnl_proj': round(pnl_total, 2)
                        }
            return best_local_pnl, best_local_params

        # STAGE 1: Coarse Grid
        coarse_alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        best_pnl, best_params = run_sweep(coarse_alphas, internal_mixes)
        
        if not best_params:
             return {'alpha': 0.1, 'weights': {'catboost':0.7, 'neural':0,'poisson':0.3}, 'pnl_proj': 0}
             
        # STAGE 2: Hyper-Fine Simplex Zoom
        # 1. Alpha Zoom: +/- 0.05 around winner, per 0.001 (0.1%)
        center_alpha = best_params['alpha']
        start_a = max(0.001, center_alpha - 0.05)
        end_a = min(0.999, center_alpha + 0.05)
        fine_alphas = np.arange(start_a, end_a + 0.0001, 0.001)
        
        # 2. Weights Zoom (Simplex Jitter)
        # Generate random variations around the best weights
        # We generate random noise and project back to simplex (sum=1)
        best_w = best_params['weights']
        base_w = np.array([best_w['catboost'], best_w['neural'], best_w['poisson']])
        
        fine_weights = [tuple(base_w)] # Keep original
        
        # Generate 200 random variations within radius
        for _ in range(200):
            # Add noise
            noise = np.random.normal(0, 0.1, 3) # Std dev 0.1
            new_w = base_w + noise
            # Clip 0-1
            new_w = np.clip(new_w, 0.0, 1.0)
            # Normalize to sum 1
            s = new_w.sum()
            if s > 0:
                new_w /= s
                if not self.has_neural:
                    # Enforce Neural=0 if strictly disabled logic preferred
                    # But here we let it float if it exists in data 0s, result is same.
                    pass
                fine_weights.append(tuple(new_w))

        best_pnl_fine, best_params_fine = run_sweep(fine_alphas, fine_weights)
        
        if best_pnl_fine > best_pnl:
            logger.info(f"Hyper Zoom: {best_params['alpha']} -> {best_params_fine['alpha']} | Weights Adjusted | PnL {best_pnl:.1f} -> {best_pnl_fine:.1f}")
            return best_params_fine
        
        return best_params

    def _save_config(self, config):
        out_path = Path('models') / 'league_weights.json'
        with open(out_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Optimization Complete. Config saved to {out_path}")

if __name__ == "__main__":
    optimizer = MetaOptimizer()
    optimizer.load_data()
    optimizer.optimize()
