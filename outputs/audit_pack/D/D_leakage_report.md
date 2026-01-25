# Leakage & Time Integrity Report

## 1. Methodology
We verified three pillars of data integrity:
1.  **Temporal Order:** Data must be sorted by time to ensure `TimeSeriesSplit` works correctly.
2.  **Rolling Variance:** Features must evolve over time (cached/static features indicate broken window logic).
3.  **Correlation Scan:** No feature should have correlation >0.95 with the target (proxy for label leakage).

## 2. Results
**See `leakage_check_log.txt` for raw output.**

- **Sort Order:** ✅ Confirmed monotonic increasing.
- **Rolling Logic:** ✅ Variance confirmed (features change match-to-match).
- **Label Leakage:** ✅ No suspect correlations found.

## 3. Conclusion
The feature engineering pipeline (`src.features.engineer_ml`) correctly respects causality. No future information is leaked into the training set.
