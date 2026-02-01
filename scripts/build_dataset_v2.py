#!/usr/bin/env python3
"""
Dataset Builder V2 (Tasks A, B, C)

Builds training dataset with:
- Correct 1X2 mapping (Draw/Tie/X recognition)
- ML odds line (Pinnacle-first, median fallback)
- Strict feature contract (28 features, 100% train/inference alignment)

Events are SKIPPED if:
- Missing valid H/D/A outcomes
- No valid ML odds line can be constructed
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ml_odds_builder import (
    build_ml_odds_line_from_normalized_rows,
    classify_outcome,
    MLOddsLine
)
from src.models.feature_contract import load_contract

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"


class EloCalculator:
    """Calculate ELO ratings from match history."""
    
    def __init__(self, k_factor: float = 20, home_advantage: float = 100):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.ratings: Dict[str, float] = {}
        self.default_rating = 1500.0
    
    def get_rating(self, team: str) -> float:
        return self.ratings.get(team, self.default_rating)
    
    def update(self, home_team: str, away_team: str, result: str):
        """Update ratings after a match."""
        home_elo = self.get_rating(home_team)
        away_elo = self.get_rating(away_team)
        
        # Expected scores with home advantage
        exp_home = 1 / (1 + 10 ** ((away_elo - home_elo - self.home_advantage) / 400))
        exp_away = 1 - exp_home
        
        # Actual scores
        if result == 'H':
            actual_home, actual_away = 1.0, 0.0
        elif result == 'A':
            actual_home, actual_away = 0.0, 1.0
        else:  # Draw
            actual_home, actual_away = 0.5, 0.5
        
        # Update ratings
        new_home = home_elo + self.k_factor * (actual_home - exp_home)
        new_away = away_elo + self.k_factor * (actual_away - exp_away)
        
        self.ratings[home_team] = new_home
        self.ratings[away_team] = new_away


class FormTracker:
    """Track team form (last 5 matches)."""
    
    def __init__(self):
        self.history: Dict[str, List[Dict]] = {}
        self.last_match_date: Dict[str, datetime] = {}
    
    def get_form(self, team: str, match_date: Optional[datetime] = None) -> Dict[str, float]:
        """Get form features for a team."""
        if team not in self.history or len(self.history[team]) == 0:
            return {
                'pts': 1.5,  # 1.5 points per game average
                'gf': 1.5,
                'ga': 1.3,
            }
        
        recent = self.history[team][-5:]
        n = len(recent)
        
        return {
            'pts': sum(m['pts'] for m in recent) / n,
            'gf': sum(m['gf'] for m in recent) / n,
            'ga': sum(m['ga'] for m in recent) / n,
        }
    
    def get_rest_days(self, team: str, match_date: datetime) -> float:
        """Get days since last match."""
        if team not in self.last_match_date:
            return 7.0  # Default
        
        delta = match_date - self.last_match_date[team]
        return max(1.0, min(30.0, delta.days))  # Clamp to 1-30 days
    
    def update(self, team: str, pts: int, gf: int, ga: int, match_date: datetime):
        """Update form after a match."""
        if team not in self.history:
            self.history[team] = []
        
        self.history[team].append({'pts': pts, 'gf': gf, 'ga': ga})
        self.last_match_date[team] = match_date


def build_features_from_match(
    row: pd.Series,
    ml_line: MLOddsLine,
    elo: EloCalculator,
    form: FormTracker,
    match_date: datetime
) -> Dict[str, Any]:
    """
    Build feature dict for a single match.
    Uses canonical feature names from contract.
    """
    home_team = str(row['HomeTeam'])
    away_team = str(row['AwayTeam'])
    league = str(row.get('Div', row.get('League', 'Unknown')))
    
    # ELO features
    elo_home = elo.get_rating(home_team)
    elo_away = elo.get_rating(away_team)
    
    # Form features
    form_home = form.get_form(home_team, match_date)
    form_away = form.get_form(away_team, match_date)
    
    # Rest days
    rest_home = form.get_rest_days(home_team, match_date)
    rest_away = form.get_rest_days(away_team, match_date)
    
    # Build feature dict with canonical names
    features = {
        # Market features from ML odds line
        'odds_home': ml_line.home_odds,
        'odds_draw': ml_line.draw_odds,
        'odds_away': ml_line.away_odds,
        'implied_home': ml_line.implied_home,
        'implied_draw': ml_line.implied_draw,
        'implied_away': ml_line.implied_away,
        'no_vig_home': ml_line.no_vig_home,
        'no_vig_draw': ml_line.no_vig_draw,
        'no_vig_away': ml_line.no_vig_away,
        'market_overround': ml_line.overround,
        'line_dispersion_home': ml_line.dispersion_home,
        'line_dispersion_draw': ml_line.dispersion_draw,
        'line_dispersion_away': ml_line.dispersion_away,
        'book_count': float(ml_line.book_count),
        
        # ELO features
        'elo_home': elo_home,
        'elo_away': elo_away,
        'elo_diff': elo_home - elo_away,
        
        # Form features
        'form_pts_home_l5': form_home['pts'],
        'form_pts_away_l5': form_away['pts'],
        'form_gf_home_l5': form_home['gf'],
        'form_gf_away_l5': form_away['gf'],
        'form_ga_home_l5': form_home['ga'],
        'form_ga_away_l5': form_away['ga'],
        
        # Rest days
        'rest_days_home': rest_home,
        'rest_days_away': rest_away,
        
        # Categorical
        'league': league,
        'home_team': home_team,
        'away_team': away_team,
    }
    
    return features


def get_result_from_score(fthg: int, ftag: int) -> str:
    """Get result from score."""
    if fthg > ftag:
        return 'H'
    elif fthg < ftag:
        return 'A'
    return 'D'


def result_to_label(result: str) -> int:
    """Convert result to numeric label."""
    return {'H': 0, 'D': 1, 'A': 2}[result]


def build_ml_line_from_historical_odds(row: pd.Series) -> Optional[MLOddsLine]:
    """
    Build ML odds line from historical match data.
    Uses available odds columns (B365, PS, Max, Avg).
    
    For historical data, we simulate the median consensus approach.
    """
    # Collect available odds
    sources = []
    
    # Bet365
    if 'B365H' in row and 'B365D' in row and 'B365A' in row:
        h, d, a = row.get('B365H'), row.get('B365D'), row.get('B365A')
        if pd.notna(h) and pd.notna(d) and pd.notna(a):
            sources.append(('bet365', float(h), float(d), float(a)))
    
    # Pinnacle
    if 'PSH' in row and 'PSD' in row and 'PSA' in row:
        h, d, a = row.get('PSH'), row.get('PSD'), row.get('PSA')
        if pd.notna(h) and pd.notna(d) and pd.notna(a):
            sources.append(('pinnacle', float(h), float(d), float(a)))
    
    # IWH/D/A (Interwetten)
    if 'IWH' in row and 'IWD' in row and 'IWA' in row:
        h, d, a = row.get('IWH'), row.get('IWD'), row.get('IWA')
        if pd.notna(h) and pd.notna(d) and pd.notna(a):
            sources.append(('interwetten', float(h), float(d), float(a)))
    
    # Max odds
    if 'MaxH' in row and 'MaxD' in row and 'MaxA' in row:
        h, d, a = row.get('MaxH'), row.get('MaxD'), row.get('MaxA')
        if pd.notna(h) and pd.notna(d) and pd.notna(a):
            sources.append(('max', float(h), float(d), float(a)))
    
    # Avg odds
    if 'AvgH' in row and 'AvgD' in row and 'AvgA' in row:
        h, d, a = row.get('AvgH'), row.get('AvgD'), row.get('AvgA')
        if pd.notna(h) and pd.notna(d) and pd.notna(a):
            sources.append(('avg', float(h), float(d), float(a)))
    
    if not sources:
        return None
    
    # Pinnacle-first strategy
    source = "median"
    pinnacle_line = next((s for s in sources if s[0] == 'pinnacle'), None)
    
    if pinnacle_line:
        home_odds, draw_odds, away_odds = pinnacle_line[1], pinnacle_line[2], pinnacle_line[3]
        source = "pinnacle"
    else:
        # Median consensus
        home_list = [s[1] for s in sources]
        draw_list = [s[2] for s in sources]
        away_list = [s[3] for s in sources]
        
        home_odds = float(np.median(home_list))
        draw_odds = float(np.median(draw_list))
        away_odds = float(np.median(away_list))
    
    # Validate odds
    if home_odds <= 1.0 or draw_odds <= 1.0 or away_odds <= 1.0:
        return None
    
    # Compute derived values
    book_count = len(sources)
    
    home_list = [s[1] for s in sources]
    draw_list = [s[2] for s in sources]
    away_list = [s[3] for s in sources]
    
    dispersion_home = float(np.std(home_list)) if len(home_list) > 1 else 0.0
    dispersion_draw = float(np.std(draw_list)) if len(draw_list) > 1 else 0.0
    dispersion_away = float(np.std(away_list)) if len(away_list) > 1 else 0.0
    
    implied_home = 1.0 / home_odds
    implied_draw = 1.0 / draw_odds
    implied_away = 1.0 / away_odds
    overround = implied_home + implied_draw + implied_away
    
    return MLOddsLine(
        home_odds=home_odds,
        draw_odds=draw_odds,
        away_odds=away_odds,
        source=source,
        book_count=book_count,
        overround=overround,
        dispersion_home=dispersion_home,
        dispersion_draw=dispersion_draw,
        dispersion_away=dispersion_away,
        no_vig_home=implied_home / overround,
        no_vig_draw=implied_draw / overround,
        no_vig_away=implied_away / overround,
        implied_home=implied_home,
        implied_draw=implied_draw,
        implied_away=implied_away
    )


def build_dataset_v2(input_file: Path, output_file: Path) -> Dict[str, Any]:
    """
    Build V2 training dataset with corrected pipeline.
    
    Returns:
        Stats dict with counts and examples
    """
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    
    # Parse dates
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
    
    # Sort by date for temporal consistency
    df = df.sort_values('Date').reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} matches")
    
    # Initialize trackers
    elo = EloCalculator()
    form = FormTracker()
    
    # Load feature contract for validation
    contract = load_contract()
    logger.info(f"Feature contract loaded: {contract.feature_count} features")
    
    # Pre-compute all ELO ratings chronologically
    logger.info("Pre-computing ELO ratings...")
    elo_snapshots: Dict[int, Tuple[float, float]] = {}
    
    for idx, row in df.iterrows():
        home = str(row['HomeTeam'])
        away = str(row['AwayTeam'])
        
        # Save ELO before update
        elo_snapshots[idx] = (elo.get_rating(home), elo.get_rating(away))
        
        # Update ELO
        try:
            fthg = int(row['FTHG'])
            ftag = int(row['FTAG'])
            result = get_result_from_score(fthg, ftag)
            elo.update(home, away, result)
        except:
            pass
    
    # Build dataset
    records = []
    stats = {
        'total': len(df),
        'processed': 0,
        'skipped_no_odds': 0,
        'skipped_invalid_result': 0,
        'skipped_examples': [],
        'pinnacle_lines': 0,
        'median_lines': 0,
    }
    
    logger.info("Building features...")
    
    for idx, row in df.iterrows():
        # Build ML odds line
        ml_line = build_ml_line_from_historical_odds(row)
        
        if ml_line is None:
            stats['skipped_no_odds'] += 1
            if len(stats['skipped_examples']) < 3:
                stats['skipped_examples'].append({
                    'idx': int(idx),
                    'home': str(row.get('HomeTeam', '')),
                    'away': str(row.get('AwayTeam', '')),
                    'reason': 'no_valid_ml_odds_line'
                })
            continue
        
        # Get result
        try:
            fthg = int(row['FTHG'])
            ftag = int(row['FTAG'])
            result = get_result_from_score(fthg, ftag)
            label = result_to_label(result)
        except:
            stats['skipped_invalid_result'] += 1
            continue
        
        # Get ELO snapshot for this match
        elo_home, elo_away = elo_snapshots[idx]
        
        # Temporarily set ELO for feature extraction
        temp_elo = EloCalculator()
        temp_elo.ratings[str(row['HomeTeam'])] = elo_home
        temp_elo.ratings[str(row['AwayTeam'])] = elo_away
        
        # Get match date for form
        match_date = pd.to_datetime(row['Date'])
        
        # Build features
        features = build_features_from_match(row, ml_line, temp_elo, form, match_date)
        features['label'] = label
        features['kickoff_time'] = row['Date']
        
        records.append(features)
        stats['processed'] += 1
        
        if ml_line.source == 'pinnacle':
            stats['pinnacle_lines'] += 1
        else:
            stats['median_lines'] += 1
        
        # Update form for next iteration
        home = str(row['HomeTeam'])
        away = str(row['AwayTeam'])
        
        home_pts = 3 if result == 'H' else (1 if result == 'D' else 0)
        away_pts = 3 if result == 'A' else (1 if result == 'D' else 0)
        
        form.update(home, home_pts, fthg, ftag, match_date)
        form.update(away, away_pts, ftag, fthg, match_date)
    
    # Create DataFrame
    result_df = pd.DataFrame(records)
    
    # Validate against feature contract
    feature_cols = contract.features
    for col in feature_cols:
        if col not in result_df.columns:
            logger.warning(f"Missing feature in output: {col}")
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_file, index=False)
    
    logger.info(f"Saved {len(result_df)} samples to {output_file}")
    logger.info(f"  Pinnacle lines: {stats['pinnacle_lines']}")
    logger.info(f"  Median lines: {stats['median_lines']}")
    logger.info(f"  Skipped (no odds): {stats['skipped_no_odds']}")
    
    return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Build V2 training dataset")
    parser.add_argument('--input', type=str, 
                        default=str(DATA_DIR / "multi_league_features_2021_2024.csv"),
                        help="Input CSV file")
    parser.add_argument('--output', type=str,
                        default=str(DATA_DIR / "ml_dataset_v2.csv"),
                        help="Output CSV file")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("DATASET BUILDER V2")
    print("=" * 60)
    
    stats = build_dataset_v2(Path(args.input), Path(args.output))
    
    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print(f"Total matches: {stats['total']}")
    print(f"Processed: {stats['processed']}")
    print(f"Skipped (no odds): {stats['skipped_no_odds']}")
    print(f"Pinnacle lines: {stats['pinnacle_lines']}")
    print(f"Median lines: {stats['median_lines']}")
    
    if stats['skipped_examples']:
        print("\nSkipped examples:")
        for ex in stats['skipped_examples']:
            print(f"  {ex['home']} vs {ex['away']}: {ex['reason']}")


if __name__ == "__main__":
    main()
