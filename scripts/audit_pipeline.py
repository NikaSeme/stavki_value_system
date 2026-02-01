#!/usr/bin/env python3
"""
Audit Pipeline Script (Task E)

Runs comprehensive checks on the corrected ML pipeline:
1. Outcome mapping (Draw/Tie/X recognition, skipped events)
2. ML odds line construction (Pinnacle-first, diagnostics)
3. Feature contract enforcement (train == live == 28 features)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ml_odds_builder import (
    classify_outcome,
    is_draw_outcome,
    build_ml_odds_line,
    MLOddsLine,
    DRAW_NAMES
)
from src.models.feature_contract import load_contract

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"


def audit_outcome_mapping():
    """
    Audit Task A: 1X2 outcome mapping.
    """
    print("\n" + "=" * 70)
    print("AUDIT A: 1X2 OUTCOME MAPPING")
    print("=" * 70)
    
    # Test draw recognition
    print("\n📋 Draw name recognition:")
    test_names = ["Draw", "draw", "DRAW", "Tie", "tie", "X", "x", "Home Win", "Away"]
    
    for name in test_names:
        result = is_draw_outcome(name)
        status = "✅ DRAW" if result else "❌ NOT draw"
        print(f"  '{name}' → {status}")
    
    print(f"\n  Recognized draw names: {sorted(DRAW_NAMES)}")
    
    # Test outcome classification
    print("\n📋 Outcome classification:")
    test_cases = [
        ("Manchester United", "Manchester United", "Liverpool", "H"),
        ("Liverpool", "Manchester United", "Liverpool", "A"),
        ("Draw", "Manchester United", "Liverpool", "D"),
        ("Tie", "Arsenal", "Chelsea", "D"),
        ("X", "Real Madrid", "Barcelona", "D"),
        ("Man Utd", "Manchester United", "Chelsea", None),  # Partial match - depends on fuzzy
    ]
    
    for outcome, home, away, expected in test_cases:
        result = classify_outcome(outcome, home, away)
        status = "✅" if result == expected else "⚠️"
        print(f"  {status} classify('{outcome}', H='{home}', A='{away}') → {result} (expected: {expected})")
    
    # Count skipped events in dataset
    print("\n📊 Skipped events analysis (from build log):")
    
    # Load historical data to analyze
    data_file = DATA_DIR / "multi_league_features_2021_2024.csv"
    if data_file.exists():
        df = pd.read_csv(data_file)
        
        # Check for missing odds
        odds_cols = ['B365H', 'B365D', 'B365A', 'PSH', 'PSD', 'PSA']
        existing_cols = [c for c in odds_cols if c in df.columns]
        
        if existing_cols:
            # Count rows with at least one complete odds set
            has_b365 = df[['B365H', 'B365D', 'B365A']].notna().all(axis=1) if all(c in df.columns for c in ['B365H', 'B365D', 'B365A']) else pd.Series([False] * len(df))
            has_ps = df[['PSH', 'PSD', 'PSA']].notna().all(axis=1) if all(c in df.columns for c in ['PSH', 'PSD', 'PSA']) else pd.Series([False] * len(df))
            
            has_valid_line = has_b365 | has_ps
            skipped = (~has_valid_line).sum()
            total = len(df)
            
            print(f"  Total matches: {total}")
            print(f"  With valid H/D/A line: {has_valid_line.sum()}")
            print(f"  Missing odds (skipped): {skipped} ({100*skipped/total:.1f}%)")
            
            # Show examples
            if skipped > 0:
                examples = df[~has_valid_line].head(3)
                print("\n  Example skipped events:")
                for _, row in examples.iterrows():
                    print(f"    {row.get('HomeTeam', 'N/A')} vs {row.get('AwayTeam', 'N/A')}")
                    print(f"      B365: H={row.get('B365H', 'N/A')}, D={row.get('B365D', 'N/A')}, A={row.get('B365A', 'N/A')}")
        else:
            print("  ⚠️ No odds columns found in raw data")
    else:
        print(f"  ⚠️ Data file not found: {data_file}")
    
    print("\n✅ Audit A complete.")


def audit_ml_odds_line():
    """
    Audit Task B: ML odds line construction.
    """
    print("\n" + "=" * 70)
    print("AUDIT B: ML ODDS LINE CONSTRUCTION")
    print("=" * 70)
    
    # Check if we have raw odds data
    raw_dir = Path(__file__).parent.parent / "outputs" / "odds"
    raw_files = list(raw_dir.glob("raw_*.json")) if raw_dir.exists() else []
    
    if raw_files:
        print(f"\n📁 Found {len(raw_files)} raw odds files")
        
        # Load most recent
        latest_file = sorted(raw_files)[-1]
        print(f"  Analyzing: {latest_file.name}")
        
        with open(latest_file) as f:
            events = json.load(f)
        
        print(f"  Events in file: {len(events)}")
        
        # Analyze 3 sample events
        print("\n📊 Sample ML odds lines:")
        samples_shown = 0
        
        for event in events[:10]:
            ml_line = build_ml_odds_line(event)
            if ml_line and samples_shown < 3:
                home = event['home_team']
                away = event['away_team']
                
                print(f"\n  {home} vs {away}:")
                print(f"    Source: {ml_line.source}")
                print(f"    Books available: {ml_line.book_count}")
                print(f"    ML Line: H={ml_line.home_odds:.2f}, D={ml_line.draw_odds:.2f}, A={ml_line.away_odds:.2f}")
                print(f"    Overround: {ml_line.overround:.4f}")
                print(f"    No-vig: H={ml_line.no_vig_home:.3f}, D={ml_line.no_vig_draw:.3f}, A={ml_line.no_vig_away:.3f}")
                print(f"    Dispersion: H={ml_line.dispersion_home:.3f}, D={ml_line.dispersion_draw:.3f}, A={ml_line.dispersion_away:.3f}")
                
                # Show bookmaker list
                bk_names = [bk['title'] for bk in event.get('bookmakers', [])]
                print(f"    Bookmakers: {', '.join(bk_names[:5])}{'...' if len(bk_names) > 5 else ''}")
                
                samples_shown += 1
        
        # Count Pinnacle vs Median lines
        pinnacle_count = 0
        median_count = 0
        skipped_count = 0
        
        for event in events:
            ml_line = build_ml_odds_line(event)
            if ml_line:
                if ml_line.source == 'pinnacle':
                    pinnacle_count += 1
                else:
                    median_count += 1
            else:
                skipped_count += 1
        
        print(f"\n  📈 Line source distribution:")
        print(f"    Pinnacle: {pinnacle_count}")
        print(f"    Median: {median_count}")
        print(f"    Skipped: {skipped_count}")
    else:
        print("\n  ⚠️ No raw odds files found. Checking historical data...")
        
        # Analyze from historical CSV
        data_file = DATA_DIR / "multi_league_features_2021_2024.csv"
        if data_file.exists():
            df = pd.read_csv(data_file)
            
            # Count Pinnacle availability
            if 'PSH' in df.columns:
                has_pinnacle = df[['PSH', 'PSD', 'PSA']].notna().all(axis=1)
                print(f"  Matches with Pinnacle odds: {has_pinnacle.sum()} / {len(df)}")
            else:
                print("  ⚠️ No Pinnacle columns (PSH/PSD/PSA) in data")
    
    print("\n✅ Audit B complete.")


def audit_feature_contract():
    """
    Audit Task C: Feature contract enforcement.
    """
    print("\n" + "=" * 70)
    print("AUDIT C: FEATURE CONTRACT ENFORCEMENT")
    print("=" * 70)
    
    # Load contract
    try:
        contract = load_contract()
        print(f"\n📋 Feature Contract v{contract.version}")
        print(f"  Total features: {contract.feature_count}")
        print(f"  Categorical: {len(contract.categorical)}")
        print(f"  Numeric: {len(contract.numeric)}")
    except Exception as e:
        print(f"  ❌ Failed to load contract: {e}")
        return
    
    # Show canonical features
    print("\n  Canonical features (first 20):")
    for i, feat in enumerate(contract.features[:20]):
        cat_mark = "(cat)" if feat in contract.categorical else ""
        print(f"    {i+1:2d}. {feat} {cat_mark}")
    if len(contract.features) > 20:
        print(f"    ... and {len(contract.features) - 20} more")
    
    # Check if V2 dataset exists
    v2_dataset = DATA_DIR / "ml_dataset_v2.csv"
    if v2_dataset.exists():
        df = pd.read_csv(v2_dataset)
        
        print(f"\n📊 V2 Dataset Analysis:")
        print(f"  Samples: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        
        # Compare with contract
        actual_features = set(df.columns) - {'label', 'kickoff_time'}
        expected_features = contract.feature_set
        
        common = actual_features & expected_features
        missing = expected_features - actual_features
        extra = actual_features - expected_features
        
        print(f"\n  Feature alignment:")
        print(f"    Contract: {len(expected_features)}")
        print(f"    Dataset:  {len(actual_features)}")
        print(f"    Common:   {len(common)}")
        print(f"    Missing:  {len(missing)} {list(missing)[:5] if missing else '✅ None'}")
        print(f"    Extra:    {len(extra)} {list(extra)[:5] if extra else '✅ None'}")
        
        if len(missing) == 0 and len(extra) == 0:
            print("\n  ✅ PERFECT MATCH: Train features == Contract features")
        else:
            print("\n  ⚠️ MISMATCH DETECTED")
        
        # Sample row
        print("\n  Sample feature vector (first row):")
        sample = df.iloc[0]
        for i, feat in enumerate(contract.features[:15]):
            if feat in df.columns:
                val = sample[feat]
                val_str = f"{val:.3f}" if isinstance(val, float) else str(val)[:20]
                print(f"    {feat:30s} = {val_str}")
    else:
        print(f"\n  ⚠️ V2 dataset not found: {v2_dataset}")
        print("  Run: python scripts/build_dataset_v2.py")
    
    # Check old v1 features for comparison
    v1_meta = MODELS_DIR / "metadata_v1_20260131_201454.json"
    if v1_meta.exists():
        with open(v1_meta) as f:
            old_meta = json.load(f)
        
        old_features = set(old_meta.get('features', []))
        
        print(f"\n📊 Old Model (V1) Comparison:")
        print(f"  V1 features: {len(old_features)}")
        print(f"  V2 features: {contract.feature_count}")
        
        v1_only = old_features - contract.feature_set
        v2_only = contract.feature_set - old_features
        
        print(f"  V1-only (removed): {len(v1_only)}")
        if v1_only:
            for f in sorted(v1_only)[:10]:
                print(f"    - {f}")
        
        print(f"  V2-only (new): {len(v2_only)}")
        if v2_only:
            for f in sorted(v2_only)[:10]:
                print(f"    + {f}")
    
    print("\n✅ Audit C complete.")


def audit_model_metrics():
    """
    Audit Task D: Model metrics comparison.
    """
    print("\n" + "=" * 70)
    print("AUDIT D: MODEL METRICS")
    print("=" * 70)
    
    # V1 metrics
    v1_meta = MODELS_DIR / "metadata_v1_20260131_201454.json"
    v2_meta = MODELS_DIR / "catboost_v2_metadata.json"
    
    if v1_meta.exists():
        with open(v1_meta) as f:
            old = json.load(f)
        
        print(f"\n📊 V1 Model (Old):")
        print(f"  Features: {old.get('num_features', 'N/A')}")
        if 'metrics' in old and 'test' in old['metrics']:
            m = old['metrics']['test']
            print(f"  Test Accuracy: {m.get('accuracy', 0):.4f}")
            print(f"  Test LogLoss: {m.get('log_loss', m.get('logloss', 0)):.4f}")
    else:
        print(f"\n  ⚠️ V1 metadata not found")
    
    if v2_meta.exists():
        with open(v2_meta) as f:
            new = json.load(f)
        
        print(f"\n📊 V2 Model (New):")
        print(f"  Features: {new.get('num_features', 'N/A')}")
        if 'metrics' in new and 'test' in new['metrics']:
            m = new['metrics']['test']
            print(f"  Test Accuracy: {m.get('accuracy', 0):.4f}")
            print(f"  Test LogLoss: {m.get('logloss', 0):.4f}")
            print(f"  Test Brier: {m.get('brier', 0):.4f}")
        
        print(f"  Calibration used: {new.get('calibration_used', 'N/A')}")
        
        if 'splits' in new:
            print(f"\n  Data splits:")
            for split, info in new['splits'].items():
                print(f"    {split}: {info['count']} samples ({info['start'][:10]} to {info['end'][:10]})")
    else:
        print(f"\n  ⚠️ V2 metadata not found. Run: python scripts/train_catboost_v2.py")
    
    print("\n✅ Audit D complete.")


def main():
    print("\n" + "=" * 70)
    print("🔍 ML PIPELINE AUDIT")
    print("=" * 70)
    print("Running comprehensive checks on corrected pipeline...")
    
    audit_outcome_mapping()
    audit_ml_odds_line()
    audit_feature_contract()
    audit_model_metrics()
    
    print("\n" + "=" * 70)
    print("✅ AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
