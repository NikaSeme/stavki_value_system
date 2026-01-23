"""
End-to-end pipeline for STAVKI betting system.

Integrates: Features → Ensemble → Calibration → EV → Staking → Recommendations
"""

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ..logging_setup import get_logger
from ..models.ensemble import EnsembleModel
from ..models.ml_model import MLModel
from ..models.poisson_model import PoissonModel
from ..strategy.ev import calculate_ev
from ..strategy.staking import kelly_stake

logger = get_logger(__name__)


def run_pipeline(
    features_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    bankroll: float = 1000.0,
    kelly_fraction: float = 0.5,
    max_stake_fraction: float = 0.05,
    ev_threshold: float = 0.0
) -> pd.DataFrame:
    """
    Run end-to-end betting pipeline.
    
    Args:
        features_df: DataFrame with features for prediction
        odds_df: DataFrame with bookmaker odds (must have match_id)
        bankroll: Total bankroll for staking
        kelly_fraction: Kelly fraction (default: 0.5 = half Kelly)
        max_stake_fraction: Max stake as fraction of bankroll
        ev_threshold: Minimum EV to include bet
        
    Returns:
        DataFrame with recommendations
    """
    logger.info("=" * 60)
    logger.info("Starting STAVKI End-to-End Pipeline")
    logger.info("=" * 60)
    
    # Step 1: Merge features and odds
    logger.info(f"Step 1: Merging {len(features_df)} features with {len(odds_df)} odds")
    
    # Ensure odds_df has match_id
    if 'match_id' not in odds_df.columns:
        odds_df = odds_df.reset_index()
        if 'index' in odds_df.columns:
            odds_df = odds_df.rename(columns={'index': 'match_id'})
    
    # Merge on match_id if available, otherwise by index
    if 'match_id' in features_df.columns and 'match_id' in odds_df.columns:
        df = features_df.merge(odds_df, on='match_id', how='inner')
    else:
        # Merge by index
        df = features_df.copy()
        for col in ['odds_home', 'odds_draw', 'odds_away']:
            if col in odds_df.columns:
                df[col] = odds_df[col].values[:len(df)]
    
    logger.info(f"Merged dataset: {len(df)} matches")
    
    # Step 2: Get probabilities (simplified - using Poisson as example)
    logger.info("Step 2: Generating predictions...")
    
    # For now, use Poisson model
    # In production, you'd use trained ensemble model
    poisson_model = PoissonModel()
    
    try:
        predictions = poisson_model.predict_dataset(df)
        
        # Copy probability columns
        df['prob_home'] = predictions['prob_home']
        df['prob_draw'] = predictions['prob_draw']
        df['prob_away'] = predictions['prob_away']
        
        logger.info("✓ Predictions generated")
    except Exception as e:
        logger.warning(f"Poisson prediction failed: {e}, using uniform probabilities")
        df['prob_home'] = 0.33
        df['prob_draw'] = 0.34
        df['prob_away'] = 0.33
    
    # Step 3: Calculate EV for each outcome
    logger.info("Step 3: Calculating Expected Value...")
    
    for outcome in ['home', 'draw', 'away']:
        prob_col = f'prob_{outcome}'
        odds_col = f'odds_{outcome}'
        ev_col = f'ev_{outcome}'
        
        if odds_col in df.columns:
            df[ev_col] = calculate_ev(df[prob_col], df[odds_col])
        else:
            logger.warning(f"Missing {odds_col}, setting EV to 0")
            df[ev_col] = 0.0
    
    logger.info("✓ EV calculated")
    
    # Step 4: Calculate stakes
    logger.info(f"Step 4: Calculating stakes (bankroll={bankroll}, Kelly={kelly_fraction})...")
    
    for outcome in ['home', 'draw', 'away']:
        prob_col = f'prob_{outcome}'
        odds_col = f'odds_{outcome}'
        stake_col = f'stake_{outcome}'
        
        if odds_col in df.columns:
            df[stake_col] = kelly_stake(
                df[prob_col],
                df[odds_col],
                bankroll=bankroll,
                kelly_fraction=kelly_fraction,
                max_stake_fraction=max_stake_fraction
            )
        else:
            df[stake_col] = 0.0
    
    # Calculate total stake per match
    df['total_stake'] = df[['stake_home', 'stake_draw', 'stake_away']].sum(axis=1)
    
    logger.info(f"✓ Stakes calculated: total={df['total_stake'].sum():.2f}")
    
    # Step 5: Filter by EV threshold
    if ev_threshold > 0:
        # Check if any outcome has positive EV
        has_positive_ev = (
            (df['ev_home'] >= ev_threshold) |
            (df['ev_draw'] >= ev_threshold) |
            (df['ev_away'] >= ev_threshold)
        )
        
        df = df[has_positive_ev].copy()
        logger.info(f"Step 5: Filtered to {len(df)} matches with EV >= {ev_threshold:.1%}")
    
    # Alert Manager
    try:
        from ..alerts.alert_manager import AlertManager
        alert_manager = AlertManager()
        alerts_enabled = True
    except ImportError:
        logger.warning("AlertManager not available")
        alerts_enabled = False

    # Step 6: Create recommendations
    recommendations = []
    
    for idx, row in df.iterrows():
        for outcome in ['home', 'draw', 'away']:
            ev = row[f'ev_{outcome}']
            stake = row[f'stake_{outcome}']
            
            if stake > 0:
                rec = {
                    'match_id': row.get('match_id', idx),
                    'date': row.get('date', ''),
                    'home_team': row.get('home_team', ''),
                    'away_team': row.get('away_team', ''),
                    'outcome': outcome,
                    'probability': row[f'prob_{outcome}'],
                    'odds': row.get(f'odds_{outcome}', 0),
                    'ev': ev,
                    'stake': stake,
                    'potential_profit': stake * (row.get(f'odds_{outcome}', 1) - 1),
                }
                recommendations.append(rec)
                
                # Send alert if enabled
                if alerts_enabled:
                    try:
                        alert_info = {
                            'match': f"{rec['home_team']} vs {rec['away_team']}",
                            'market': f"{outcome.title()} Win",
                            'odds': rec['odds'],
                            'model_prob': rec['probability'],
                            'ev': rec['ev'] * 100, # Convert to percent
                            'stake': rec['stake']
                        }
                        alert_manager.send_value_bet_alert(alert_info)
                    except Exception as e:
                        logger.error(f"Failed to send alert for {rec['match_id']}: {e}")
    
    recommendations_df = pd.DataFrame(recommendations)
    
    logger.info("=" * 60)
    logger.info(f"Pipeline Complete: {len(recommendations_df)} recommendations")
    if not recommendations_df.empty:
        logger.info(f"Total stake: {recommendations_df['stake'].sum():.2f}")
        logger.info(f"Avg EV: {recommendations_df['ev'].mean():.2%}")
    else:
        logger.info("No recommendations found.")
    logger.info("=" * 60)
    
    return recommendations_df


def save_recommendations(
    recommendations_df: pd.DataFrame,
    output_dir: Path,
    prefix: str = "recommendations"
) -> Dict[str, Path]:
    """
    Save recommendations to CSV and JSON.
    
    Args:
        recommendations_df: Recommendations DataFrame
        output_dir: Output directory
        prefix: Filename prefix
        
    Returns:
        Dict with paths to saved files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    # Save CSV
    csv_path = output_dir / f"{prefix}.csv"
    recommendations_df.to_csv(csv_path, index=False)
    paths['csv'] = csv_path
    logger.info(f"✓ Saved CSV: {csv_path}")
    
    # Save JSON
    json_path = output_dir / f"{prefix}.json"
    recommendations_df.to_json(json_path, orient='records', indent=2)
    paths['json'] = json_path
    logger.info(f"✓ Saved JSON: {json_path}")
    
    return paths
