"""
Backtest script for STAVKI Value System (Task L).

Runs Kelly betting simulation on historical data to compute:
- ROI (%)
- Max Drawdown (%)
- Sharpe Ratio
- Win Rate (%)
- Total Bets

Usage:
    python scripts/run_backtest.py --start 2024-01-01 --end 2024-12-31 --bankroll 1000
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging_setup import get_logger

logger = get_logger(__name__)


def kelly_fraction(prob: float, odds: float, kelly_multiplier: float = 0.25) -> float:
    """
    Calculate Kelly criterion bet fraction.
    
    Args:
        prob: Estimated probability of winning
        odds: Decimal odds offered
        kelly_multiplier: Fraction of Kelly to use (default: 0.25 = quarter Kelly)
        
    Returns:
        Fraction of bankroll to bet (0 if negative edge)
    """
    if odds <= 1.0 or prob <= 0 or prob >= 1:
        return 0.0
    
    # Kelly formula: (p * (odds - 1) - (1 - p)) / (odds - 1)
    edge = prob * odds - 1.0
    if edge <= 0:
        return 0.0
    
    kelly = edge / (odds - 1)
    return max(0.0, min(kelly * kelly_multiplier, 0.10))  # Cap at 10%


def run_backtest(
    df: pd.DataFrame,
    prob_col: str = 'prob_home',
    odds_col: str = 'home_odds',
    result_col: str = 'FTR',
    target_result: str = 'H',
    initial_bankroll: float = 1000.0,
    kelly_multiplier: float = 0.25,
    min_edge: float = 0.05,
    min_odds: float = 1.50
) -> dict:
    """
    Run backtest simulation.
    
    Args:
        df: DataFrame with predictions, odds, and results
        prob_col: Column with predicted probability
        odds_col: Column with decimal odds
        result_col: Column with actual result
        target_result: Result to bet on ('H', 'D', or 'A')
        initial_bankroll: Starting bankroll
        kelly_multiplier: Kelly fraction multiplier
        min_edge: Minimum edge required to bet
        min_odds: Minimum odds required to bet
        
    Returns:
        Dict with backtest metrics
    """
    bankroll = initial_bankroll
    peak_bankroll = initial_bankroll
    max_drawdown = 0.0
    
    bets = []
    daily_returns = []
    wins = 0
    losses = 0
    
    for _, row in df.iterrows():
        prob = row.get(prob_col, 0)
        odds = row.get(odds_col, 1.0)
        result = row.get(result_col, '')
        
        # Skip if missing data or odds too low
        if pd.isna(prob) or pd.isna(odds) or odds < min_odds:
            continue
        
        # Calculate edge
        edge = prob * odds - 1.0
        if edge < min_edge:
            continue
        
        # Calculate bet size
        fraction = kelly_fraction(prob, odds, kelly_multiplier)
        if fraction <= 0:
            continue
        
        stake = bankroll * fraction
        
        # Determine outcome
        won = (result == target_result)
        
        if won:
            profit = stake * (odds - 1)
            wins += 1
        else:
            profit = -stake
            losses += 1
        
        bankroll += profit
        bets.append({
            'date': row.get('Date'),
            'stake': stake,
            'odds': odds,
            'prob': prob,
            'edge': edge,
            'won': won,
            'profit': profit,
            'bankroll': bankroll
        })
        
        # Track drawdown
        peak_bankroll = max(peak_bankroll, bankroll)
        current_drawdown = (peak_bankroll - bankroll) / peak_bankroll
        max_drawdown = max(max_drawdown, current_drawdown)
        
        # Daily return
        if len(bets) > 0:
            daily_returns.append(profit / (bankroll - profit) if bankroll > profit else 0)
    
    # Calculate metrics
    total_bets = wins + losses
    roi = (bankroll - initial_bankroll) / initial_bankroll * 100 if initial_bankroll > 0 else 0
    win_rate = wins / total_bets * 100 if total_bets > 0 else 0
    
    # Sharpe ratio (annualized)
    if len(daily_returns) > 1:
        mean_return = np.mean(daily_returns)
        std_return = np.std(daily_returns)
        sharpe = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
    else:
        sharpe = 0.0
    
    return {
        'roi_pct': roi,
        'max_drawdown_pct': max_drawdown * 100,
        'sharpe_ratio': sharpe,
        'win_rate_pct': win_rate,
        'total_bets': total_bets,
        'wins': wins,
        'losses': losses,
        'final_bankroll': bankroll,
        'initial_bankroll': initial_bankroll,
        'bets': bets
    }


def main():
    parser = argparse.ArgumentParser(description='STAVKI Backtest')
    parser.add_argument('--start', type=str, default='2024-01-01',
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-12-31',
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--bankroll', type=float, default=1000.0,
                        help='Initial bankroll')
    parser.add_argument('--kelly', type=float, default=0.25,
                        help='Kelly multiplier (default: 0.25)')
    parser.add_argument('--min-edge', type=float, default=0.05,
                        help='Minimum edge to bet (default: 0.05)')
    parser.add_argument('--data-file', type=str, default=None,
                        help='Path to predictions CSV')
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("STAVKI BACKTEST")
    logger.info("=" * 70)
    
    # Load data
    base_dir = Path(__file__).parent.parent
    
    if args.data_file:
        data_file = Path(args.data_file)
    else:
        data_file = base_dir / 'data' / 'processed' / 'multi_league_features_2021_2024.csv'
    
    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        return
    
    df = pd.read_csv(data_file)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter by date range
    start_date = pd.to_datetime(args.start)
    end_date = pd.to_datetime(args.end)
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    logger.info(f"Date range: {args.start} to {args.end}")
    logger.info(f"Matches: {len(df)}")
    
    # Check for required columns
    required_prob_cols = ['prob_home', 'prob_draw', 'prob_away']
    required_odds_cols = ['B365H', 'B365D', 'B365A']  # Common odds columns
    
    # Try to find odds columns
    odds_map = {}
    for odds_col in ['B365H', 'BbAvH', 'PSH', 'WHH']:
        if odds_col in df.columns:
            odds_map['home_odds'] = odds_col
            break
    for odds_col in ['B365D', 'BbAvD', 'PSD', 'WHD']:
        if odds_col in df.columns:
            odds_map['draw_odds'] = odds_col
            break
    for odds_col in ['B365A', 'BbAvA', 'PSA', 'WHA']:
        if odds_col in df.columns:
            odds_map['away_odds'] = odds_col
            break
    
    if not odds_map:
        logger.warning("No odds columns found - using placeholder odds of 2.0")
        df['home_odds'] = 2.0
        df['draw_odds'] = 3.0
        df['away_odds'] = 3.5
    else:
        df['home_odds'] = df[odds_map.get('home_odds', 'B365H')]
        df['draw_odds'] = df[odds_map.get('draw_odds', 'B365D')]
        df['away_odds'] = df[odds_map.get('away_odds', 'B365A')]
    
    # Check for probability columns (if not present, we need a model)
    if 'prob_home' not in df.columns:
        logger.warning("No probability columns found - using placeholder 1/odds")
        total = 1/df['home_odds'] + 1/df['draw_odds'] + 1/df['away_odds']
        df['prob_home'] = (1/df['home_odds']) / total
        df['prob_draw'] = (1/df['draw_odds']) / total
        df['prob_away'] = (1/df['away_odds']) / total
    
    # Run backtest for each outcome
    results = {}
    for outcome, prob_col, odds_col, target in [
        ('Home', 'prob_home', 'home_odds', 'H'),
        ('Draw', 'prob_draw', 'draw_odds', 'D'),
        ('Away', 'prob_away', 'away_odds', 'A')
    ]:
        result = run_backtest(
            df, 
            prob_col=prob_col, 
            odds_col=odds_col,
            target_result=target,
            initial_bankroll=args.bankroll,
            kelly_multiplier=args.kelly,
            min_edge=args.min_edge
        )
        results[outcome] = result
    
    # Combined results
    logger.info("\n" + "=" * 70)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 70)
    
    for outcome, r in results.items():
        logger.info(f"\n{outcome}:")
        logger.info(f"  Bets:         {r['total_bets']}")
        logger.info(f"  Win Rate:     {r['win_rate_pct']:.1f}%")
        logger.info(f"  ROI:          {r['roi_pct']:+.2f}%")
        logger.info(f"  Max Drawdown: {r['max_drawdown_pct']:.1f}%")
        logger.info(f"  Sharpe:       {r['sharpe_ratio']:.2f}")
        logger.info(f"  Final Bank:   ${r['final_bankroll']:.2f}")
    
    # Overall summary
    total_bets = sum(r['total_bets'] for r in results.values())
    total_profit = sum(r['final_bankroll'] - r['initial_bankroll'] for r in results.values())
    total_initial = args.bankroll * 3
    overall_roi = total_profit / total_initial * 100
    
    logger.info("\n" + "-" * 70)
    logger.info(f"OVERALL (all outcomes):")
    logger.info(f"  Total Bets: {total_bets}")
    logger.info(f"  Total ROI:  {overall_roi:+.2f}%")
    logger.info(f"  Net Profit: ${total_profit:+.2f}")


if __name__ == '__main__':
    main()
