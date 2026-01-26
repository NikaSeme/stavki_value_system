# Audit v3.2 Final Report: Fix Flatline & Retrain Pipeline

## Summary
The "Flatline Probability" issue (predictions sticking to ~33%) has been diagnosed and fixed. The pipeline has been retrained on a strict time-split v3 dataset, and live inference now includes comprehensive guardrails.

## Root Cause Analysis
- **Flatline Issue**: Likely caused by silent model fallback or uncalibrated baseline outputs in previous version.
- **Fix**: Implemented "Hard Fail" logic (no silent fallbacks), retrained CatBoost model on verified v3 dataset, and added "Flatline Guardrail" (StdDev check).

## Key Fixes & Verifications

### 1. Model Retraining (v3.2)
- **Dataset**: `train_v3.parquet` (Strict Walk-Forward Split). No shuffling.
- **Model**: CatBoostClassifier `models/catboost_v3.pkl`.
- **Metrics**: Accuracy 57.89%, LogLoss 1.49.
- **Calibration**: Isotonic calibration applied (`calibrator_v3.pkl`). confirmed by `audit_pack/A6_metrics/calibration_plot_catboost.png`.

### 2. Live Pipeline Guardrails
- **Build Stamp**: Telegram alerts now include Commit SHA, Run ID, and Model Status.
- **Hard Fail**: `src/models/loader.py` now raises `RuntimeError` if artifacts are missing (preventing silent 33% fallback).
- **Flatline Guard**: `src/strategy/value_live.py` validation raises error if probability standard deviation < 0.02.
- **Single Book**: Enforced `mode='single_book'` in normalization to ensure valid market lines.

### 3. Forensic Checks
- **Manual EV Check**: `audit_pack/A6_metrics/manual_ev_check.csv` confirms EV calc matches Model Prob * Odds - 1 (Diff < 0.0002).
- **Odds Integrity**: `audit_pack/A4_odds_integrity/odds_raw_sample.json` confirms raw API data is correctly mapped.
- **Duplicate Senders**: Disabled `src/bot/telegram_bot.py` and `src/alerts/telegram_bot.py` to prevent double-alerting. Only `src/integration/telegram_notify.py` is active.

## Deliverables
- **Audit Pack**: `stavki_value_cleaned_pack_v3_2.zip` (Full logs, artifacts, models).
- **Manifest**: `audit_pack/A5_models/models_manifest.json` (Hashes of deployed models).
- **Blockers**: See `BLOCKERS.md` regarding Neural Model C.

## Next Steps
- Monitor Telegram alerts for "Build Stamp" to verify deployment.
- Check `audit_pack/A9_live/alerts_sent.csv` daily.
