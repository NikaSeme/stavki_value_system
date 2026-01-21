"""
Production reporting module for STAVKI betting system.

Generates comprehensive reports with warnings, bankroll tracking, and multiple formats.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..logging_setup import get_logger

logger = get_logger(__name__)


def generate_report(
    recommendations_df: pd.DataFrame,
    initial_bankroll: float,
    ev_threshold: float,
    warnings: List[str]
) -> Dict:
    """
    Generate comprehensive betting report.
    
    Args:
        recommendations_df: Recommendations DataFrame
        initial_bankroll: Initial bankroll
        ev_threshold: EV threshold used
        warnings: List of warnings
        
    Returns:
        Report dictionary
    """
    total_stake = recommendations_df['stake'].sum() if len(recommendations_df) > 0 else 0
    remaining_bankroll = initial_bankroll - total_stake
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'bankroll': {
            'initial': initial_bankroll,
            'used': total_stake,
            'remaining': remaining_bankroll,
            'utilization_pct': (total_stake / initial_bankroll * 100) if initial_bankroll > 0 else 0
        },
        'filters': {
            'ev_threshold': ev_threshold
        },
        'summary': {
            'total_bets': len(recommendations_df),
            'total_stake': total_stake,
            'avg_stake': recommendations_df['stake'].mean() if len(recommendations_df) > 0 else 0,
            'avg_ev': recommendations_df['ev'].mean() if len(recommendations_df) > 0 else 0,
            'avg_odds': recommendations_df['odds'].mean() if len(recommendations_df) > 0 else 0,
            'total_potential_profit': recommendations_df['potential_profit'].sum() if len(recommendations_df) > 0 else 0
        },
        'warnings': warnings,
        'bets': recommendations_df.to_dict(orient='records') if len(recommendations_df) > 0 else []
    }
    
    return report


def save_report_json(report: Dict, output_path: Path) -> None:
    """
    Save report as JSON.
    
    Args:
        report: Report dictionary
        output_path: Path to save JSON
    """
    import json
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"✓ Saved JSON report: {output_path}")


def save_report_txt(report: Dict, output_path: Path) -> None:
    """
    Save human-readable text report.
    
    Args:
        report: Report dictionary
        output_path: Path to save text report
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = []
    
    # Header
    lines.append("=" * 70)
    lines.append("STAVKI BETTING RECOMMENDATIONS REPORT")
    lines.append("=" * 70)
    lines.append(f"Generated: {report['timestamp']}")
    lines.append("")
    
    # Bankroll
    lines.append("BANKROLL STATUS")
    lines.append("-" * 70)
    lines.append(f"Initial:     ${report['bankroll']['initial']:>12,.2f}")
    lines.append(f"Used:        ${report['bankroll']['used']:>12,.2f}")
    lines.append(f"Remaining:   ${report['bankroll']['remaining']:>12,.2f}")
    lines.append(f"Utilization: {report['bankroll']['utilization_pct']:>12.1f}%")
    lines.append("")
    
    # Summary
    lines.append("SUMMARY")
    lines.append("-" * 70)
    lines.append(f"Total Bets:           {report['summary']['total_bets']:>6}")
    lines.append(f"Total Stake:          ${report['summary']['total_stake']:>10,.2f}")
    lines.append(f"Average Stake:        ${report['summary']['avg_stake']:>10,.2f}")
    lines.append(f"Average EV:           {report['summary']['avg_ev']:>10.2%}")
    lines.append(f"Average Odds:         {report['summary']['avg_odds']:>10.2f}")
    lines.append(f"Potential Profit:     ${report['summary']['total_potential_profit']:>10,.2f}")
    lines.append(f"EV Threshold:         {report['filters']['ev_threshold']:>10.2%}")
    lines.append("")
    
    # Warnings
    if report['warnings']:
        lines.append("WARNINGS")
        lines.append("-" * 70)
        for warning in report['warnings']:
            lines.append(f"⚠ {warning}")
        lines.append("")
    
    # Bets
    if report['bets']:
        lines.append("RECOMMENDED BETS")
        lines.append("=" * 70)
        lines.append("")
        
        for i, bet in enumerate(report['bets'], 1):
            lines.append(f"Bet #{i}")
            lines.append("-" * 70)
            lines.append(f"Match:       {bet.get('home_team', 'N/A')} vs {bet.get('away_team', 'N/A')}")
            lines.append(f"Date:        {bet.get('date', 'N/A')}")
            lines.append(f"Outcome:     {bet['outcome'].upper()}")
            lines.append(f"Probability: {bet['probability']:.1%}")
            lines.append(f"Odds:        {bet['odds']:.2f}")
            lines.append(f"EV:          {bet['ev']:.2%}")
            lines.append(f"Stake:       ${bet['stake']:.2f}")
            lines.append(f"Potential:   ${bet['potential_profit']:.2f}")
            lines.append("")
    else:
        lines.append("NO BETS RECOMMENDED")
        lines.append("-" * 70)
        lines.append("No bets meet the EV threshold criteria.")
        lines.append("")
    
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"✓ Saved text report: {output_path}")


def collect_warnings(
    features_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    recommendations_df: pd.DataFrame,
    initial_bankroll: float
) -> List[str]:
    """
    Collect warnings about data quality and recommendations.
    
    Args:
        features_df: Features DataFrame
        odds_df: Odds DataFrame
        recommendations_df: Recommendations DataFrame
        initial_bankroll: Initial bankroll
        
    Returns:
        List of warning messages
    """
    warnings = []
    
    # Check for missing odds
    if 'match_id' in features_df.columns and 'match_id' in odds_df.columns:
        features_ids = set(features_df['match_id'])
        odds_ids = set(odds_df['match_id'])
        missing_odds = features_ids - odds_ids
        
        if missing_odds:
            warnings.append(f"{len(missing_odds)} matches missing odds data")
    
    # Check bankroll usage
    if len(recommendations_df) > 0:
        total_stake = recommendations_df['stake'].sum()
        
        if total_stake > initial_bankroll:
            warnings.append(f"Total stake (${total_stake:.2f}) exceeds bankroll (${initial_bankroll:.2f})")
        
        if total_stake > initial_bankroll * 0.9:
            warnings.append(f"High bankroll utilization: {total_stake/initial_bankroll:.1%}")
    
    # Check for zero bets
    if len(recommendations_df) == 0:
        warnings.append("No bets meet the criteria (try lowering EV threshold)")
    
    return warnings
