import pandas as pd
import numpy as np
from pathlib import Path

def audit_leakage():
    print("AUDIT: Checking for Data Leakage...")
    
    # Load data
    data_path = Path("data/processed/features.csv")
    if not data_path.exists():
        print("SKIP: Feature file not found")
        return

    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 1. Check Sort Order
    is_sorted = df['Date'].is_monotonic_increasing
    print(f"CHECK 1: Date Monotonic Increasing? {'✅ YES' if is_sorted else '❌ NO'}")
    
    # 2. Check Rolling Features (Concept: Current match result shouldn't affect 'L5' features)
    # We can't easily prove this without re-calculating, but we can check logic.
    # Proxy check: Do rows with same teams have different L5 values?
    print("CHECK 2: Feature Variance (Proxy for Rolling)")
    home_vars = df.groupby('HomeTeam')['HomePointsL5'].var().mean()
    print(f"   Avg Variance in HomePointsL5: {home_vars:.2f} (Should be > 0)")
    
    # 3. Check Target Leakage
    # Target is FTR. Check correlations. If correlation is 1.0, it's leakage.
    print("CHECK 3: Target Correlation Scan")
    if 'FTR' in df.columns:
        # Encode FTR
        df['target_num'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        leaks = []
        for col in numeric_cols:
            if col == 'target_num': continue
            corr = df[col].corr(df['target_num'])
            if abs(corr) > 0.95:
                leaks.append((col, corr))
        
        if leaks:
             print(f"⚠️ POTENTIAL LEAKAGE: {leaks}")
        else:
             print("✅ No suspicious >0.95 correlations found")

if __name__ == "__main__":
    audit_leakage()
