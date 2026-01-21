"""
Evaluation and metrics module for STAVKI betting system.

Calculates performance metrics from betting results.
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..logging_setup import get_logger

logger = get_logger(__name__)


def load_results(results_path: Path) -> pd.DataFrame:
    """
    Load betting results from CSV.
    
    Expected columns:
    - match_id: Match identifier
    - market: Market type (e.g., '1X2', 'home', 'draw', 'away')
    - selection: Bet selection
    - stake: Amount staked
    - odds: Odds
    - outcome: Result ('win', 'loss', 'void')
    - payout: Amount returned (0 for loss, stake*odds for win)
    
    Args:
        results_path: Path to results CSV
        
    Returns:
        Results DataFrame
    """
    df = pd.read_csv(results_path)
    
    required_cols = ['match_id', 'stake', 'odds', 'outcome']
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Add payout if not present
    if 'payout' not in df.columns:
        df['payout'] = df.apply(
            lambda row: row['stake'] * row['odds'] if row['outcome'] == 'win' else 0.0,
            axis=1
        )
    
    return df


def calculate_metrics(results_df: pd.DataFrame) -> Dict:
    """
    Calculate betting performance metrics.
    
    Args:
        results_df: Results DataFrame
        
    Returns:
        Metrics dictionary
    """
    # Filter out void bets for metrics
    active_bets = results_df[results_df['outcome'] != 'void'].copy()
    
    if len(active_bets) == 0:
        return {
            'number_of_bets': 0,
            'total_staked': 0.0,
            'total_returned': 0.0,
            'profit': 0.0,
            'roi': 0.0,
            'hit_rate': 0.0,
            'wins': 0,
            'losses': 0,
            'voids': len(results_df)
        }
    
    # Basic counts
    num_bets = len(active_bets)
    wins = (active_bets['outcome'] == 'win').sum()
    losses = (active_bets['outcome'] == 'loss').sum()
    voids = (results_df['outcome'] == 'void').sum()
    
    # Financial metrics
    total_staked = active_bets['stake'].sum()
    total_returned = active_bets['payout'].sum()
    profit = total_returned - total_staked
    
    # ROI (Return on Investment)
    roi = (profit / total_staked * 100) if total_staked > 0 else 0.0
    
    # Hit rate (win percentage)
    hit_rate = (wins / num_bets * 100) if num_bets > 0 else 0.0
    
    # Average odds
    avg_odds = active_bets['odds'].mean() if num_bets > 0 else 0.0
    
    # Average stake
    avg_stake = active_bets['stake'].mean() if num_bets > 0 else 0.0
    
    metrics = {
        'number_of_bets': num_bets,
        'total_staked': float(total_staked),
        'total_returned': float(total_returned),
        'profit': float(profit),
        'roi': float(roi),
        'hit_rate': float(hit_rate),
        'wins': int(wins),
        'losses': int(losses),
        'voids': int(voids),
        'avg_odds': float(avg_odds),
        'avg_stake': float(avg_stake)
    }
    
    logger.info(f"Calculated metrics: {num_bets} bets, ROI={roi:.2f}%, Hit Rate={hit_rate:.2f}%")
    
    return metrics


def generate_evaluation_report(metrics: Dict, results_df: pd.DataFrame) -> str:
    """
    Generate human-readable evaluation report.
    
    Args:
        metrics: Metrics dictionary
        results_df: Results DataFrame
        
    Returns:
        Report text
    """
    lines = []
    
    lines.append("=" * 70)
    lines.append("BETTING PERFORMANCE EVALUATION REPORT")
    lines.append("=" * 70)
    lines.append("")
    
    # Overview
    lines.append("OVERVIEW")
    lines.append("-" * 70)
    lines.append(f"Total Bets:        {metrics['number_of_bets']:>10}")
    lines.append(f"Wins:              {metrics['wins']:>10}")
    lines.append(f"Losses:            {metrics['losses']:>10}")
    lines.append(f"Voids:             {metrics['voids']:>10}")
    lines.append("")
    
    # Financial Performance
    lines.append("FINANCIAL PERFORMANCE")
    lines.append("-" * 70)
    lines.append(f"Total Staked:      ${metrics['total_staked']:>12,.2f}")
    lines.append(f"Total Returned:    ${metrics['total_returned']:>12,.2f}")
    lines.append(f"Profit/Loss:       ${metrics['profit']:>12,.2f}")
    lines.append(f"ROI:               {metrics['roi']:>12.2f}%")
    lines.append("")
    
    # Performance Metrics
    lines.append("PERFORMANCE METRICS")
    lines.append("-" * 70)
    lines.append(f"Hit Rate:          {metrics['hit_rate']:>12.2f}%")
    lines.append(f"Average Odds:      {metrics['avg_odds']:>12.2f}")
    lines.append(f"Average Stake:     ${metrics['avg_stake']:>12,.2f}")
    lines.append("")
    
    # Profitability Analysis
    if metrics['profit'] > 0:
        lines.append("✓ PROFITABLE: System shows positive returns")
    elif metrics['profit'] < 0:
        lines.append("✗ LOSS: System shows negative returns")
    else:
        lines.append("○ BREAK-EVEN: No profit or loss")
    
    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)
    
    return '\n'.join(lines)


def save_evaluation_summary(
    metrics: Dict,
    report_text: str,
    output_dir: Path,
    prefix: str = "summary_report"
) -> Dict[str, Path]:
    """
    Save evaluation summary to JSON and text files.
    
    Args:
        metrics: Metrics dictionary
        report_text: Report text
        output_dir: Output directory
        prefix: Filename prefix
        
    Returns:
        Dict with paths to saved files
    """
    import json
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    # Save JSON
    json_path = output_dir / f"{prefix}.json"
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    paths['json'] = json_path
    logger.info(f"Saved evaluation JSON: {json_path}")
    
    # Save text
    txt_path = output_dir / f"{prefix}.txt"
    txt_path.write_text(report_text)
    paths['txt'] = txt_path
    logger.info(f"Saved evaluation report: {txt_path}")
    
    return paths
