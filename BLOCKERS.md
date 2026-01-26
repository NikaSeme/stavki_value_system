# BLOCKERS (Audit v3.2)

## 1. Neural Model C (Line Movement)
**Status**: Skipped / Deferred.

**Reason**:
Implementing a "Line Movement" Neural Network requires historical **odds snapshot sequences** (e.g., odds at T-24h, T-12h, T-1h for the *same* match). 
The current dataset (`epl_features_2021_2024.csv`) provides static features and closing odds, but lacks the granular time-series data of odds changes needed to train a Sequence Model (RNN/LSTM) effectively.

**Recommendation**:
To enable this feature in v4:
1. Start collecting odds snapshots immediately via `scripts/run_odds_pipeline.py` (running hourly via cron).
2. Accumulate at least 3 months of snapshot history.
3. Build a new dataset generator that aggregates these snapshots into time-series tensors.
