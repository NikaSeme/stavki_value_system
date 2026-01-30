
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
    
    def __init__(self, history_path: str = "data/processed/multi_league_features_6leagues_full.csv"):
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
                 
        logger.info(f"Loaded {len(self.df)} matches.")

        # Prepare Probabilities
        self._generate_model_probs()
        self._prepare_market_probs()
        
    def _generate_model_probs(self):
        """Generate raw model predictions for the entire dataset."""
        logger.info("Generating Model Predictions (Ensemble)...")
        from src.models.ensemble_predictor import EnsemblePredictor
        ensemble = EnsemblePredictor()
        
        # We need raw component predictions for Internal Weight Optimization
        # EnsemblePredictor returns (ensemble_probs, components_dict)
        # We need to hack it slightly or use _predict_components if exposed.
        # But predict() returns components!
        
        # Metadata DF
        meta = self.df[['home_team', 'away_team', 'commence_time']].copy()
        
        try:
            # This might be slow for 7k matches
            probs, components = ensemble.predict(meta, None, features=self.df)
            
            # Store Component Probs in DF for fast vectorization
            # components['catboost'] is (N, 3)
            # components['poisson'] is (N, 3)
            # components['neural'] is (N, 3) or None
            
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
            else:
                self.has_neural = False
                # Fill with zeros or fallback
                self.df['nn_h'] = 0; self.df['nn_d'] = 0; self.df['nn_a'] = 0
                
        except Exception as e:
            logger.error(f"Prediction generation failed: {e}")
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
        """Find best Alpha and Internal Weights for a dataframe."""
        
        # Parameter Grid
        alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        # Internal Weights Grid (Sum ~ 1.0)
        # (CatBoost, Neural, Poisson)
        if self.has_neural:
            internal_mixes = [
                (0.5, 0.3, 0.2), # Default
                (0.7, 0.2, 0.1), # Heavy CatBoost
                (0.3, 0.6, 0.1), # Heavy Neural
                (0.4, 0.4, 0.2)  # Balanced ML
            ]
        else:
             internal_mixes = [
                 (0.7, 0.0, 0.3), # Default 2-model
                 (0.5, 0.0, 0.5),
                 (0.9, 0.0, 0.1)
             ]
             
        best_pnl = -float('inf')
        best_params = None
        
        # Pre-calc columns to avoid re-access
        # ... logic to speed up ...
        
        for alpha, (wc, wn, wp) in product(alphas, internal_mixes):
            
            # 1. Calc Internal Ensemble Prob
            p_ens_h = df['cb_h']*wc + df['nn_h']*wn + df['ps_h']*wp
            p_ens_d = df['cb_d']*wc + df['nn_d']*wn + df['ps_d']*wp
            p_ens_a = df['cb_a']*wc + df['nn_a']*wn + df['ps_a']*wp
            
            # Normalize (Internal)
            s = p_ens_h + p_ens_d + p_ens_a
            p_ens_h /= s; p_ens_d /= s; p_ens_a /= s
            
            # 2. Calc Final Prob (External Alpha)
            p_fin_h = alpha * p_ens_h + (1-alpha) * df['mk_h']
            p_fin_d = alpha * p_ens_d + (1-alpha) * df['mk_d']
            p_fin_a = alpha * p_ens_a + (1-alpha) * df['mk_a']
            
            # 3. Simulate PnL
            # Bet if EV > 5%
            
            # Vectorized PnL
            # EV = (Prob * Odds) - 1
            # PnL = Sum(Odds - 1) where Win - Sum(1) where Loss
            
            pnl_total = 0
            
            # Home
            ev_h = (p_fin_h * df['AvgOddsH']) - 1
            bet_h = ev_h > 0.05
            win_h = df['FTR'] == 'H'
            pnl_total += (df.loc[bet_h & win_h, 'AvgOddsH'] - 1).sum()
            pnl_total -= (~win_h & bet_h).sum()
            
            # Draw
            ev_d = (p_fin_d * df['AvgOddsD']) - 1
            bet_d = ev_d > 0.05
            win_d = df['FTR'] == 'D'
            pnl_total += (df.loc[bet_d & win_d, 'AvgOddsD'] - 1).sum()
            pnl_total -= (~win_d & bet_d).sum()

            # Away
            ev_a = (p_fin_a * df['AvgOddsA']) - 1
            bet_a = ev_a > 0.05
            win_a = df['FTR'] == 'A'
            pnl_total += (df.loc[bet_a & win_a, 'AvgOddsA'] - 1).sum()
            pnl_total -= (~win_a & bet_a).sum()
            
            # Drawdown Check (Simple approximation: Max single loss streak cost? Or just check final PnL?)
            # For simplicity, we maximize PnL. If PnL < 0, we dislike it.
            
            if pnl_total > best_pnl:
                best_pnl = pnl_total
                best_params = {
                    'alpha': alpha,
                    'weights': {'catboost': wc, 'neural': wn, 'poisson': wp},
                    'pnl_proj': round(pnl_total, 2)
                }
                
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
