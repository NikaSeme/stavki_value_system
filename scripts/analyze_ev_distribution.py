#!/usr/bin/env python3
"""
Compare EVs: Isotonic vs Platt Scaling

This script loads both the old (isotonic) and new (Platt) ensemble models
and compares their EVs on the same data to demonstrate the improvement.
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load the latest bets
bets_file = Path('audit_pack/A9_live/top_ev_bets.csv')

if not bets_file.exists():
    print("❌ No bets file found. Run value finder first.")
    sys.exit(1)

bets = pd.read_csv(bets_file)

print("=" * 70)
print("EV DISTRIBUTION ANALYSIS (Platt Scaling)")
print("=" * 70)
print()
print(f"Total bets: {len(bets)}")
print()

# EV ranges
ranges = [
    (0, 10, "0-10%"),
    (10, 20, "10-20%"),
    (20, 30, "20-30%"),
    (30, 50, "30-50%"),
    (50, 100, "50-100%"),
    (100, 200, "100-200%"),
    (200, 1000, "200%+")
]

print("EV Distribution:")
print("-" * 70)
for low, high, label in ranges:
    count = len(bets[(bets['ev_pct'] >= low) & (bets['ev_pct'] < high)])
    pct = count / len(bets) * 100 if len(bets) > 0 else 0
    bar = "█" * int(pct / 2)
    print(f"{label:12} | {count:3} bets ({pct:5.1f}%) {bar}")

print()
print("Key Statistics:")
print(f"  Median EV: {bets['ev_pct'].median():.1f}%")
print(f"  Mean EV: {bets['ev_pct'].mean():.1f}%")
print(f"  Max EV: {bets['ev_pct'].max():.1f}%")
print()

# Count extreme EVs
extreme = len(bets[bets['ev_pct'] > 100])
if extreme > 0:
    print(f"⚠️  {extreme} bets with >100% EV ({extreme/len(bets)*100:.1f}%)")
    print("   This suggests calibration may still need adjustment")
else:
    print("✅ No bets with >100% EV")
    print("   Calibration appears reasonable")

print()
print("Top 5 EV Bets:")
print("-" * 70)
top5 = bets.nlargest(5, 'ev_pct')
for idx, row in top5.iterrows():
    print(f"  {row['selection'][:25]:25} @ {row['odds']:4.1f} → EV: {row['ev_pct']:5.1f}%")

