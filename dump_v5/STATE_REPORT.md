# STATE REPORT: Stavki Value System (v5.0)

**Date**: 2025-01-25
**Version**: v5.0 (Vienna Schedule)
**Status**: Production Ready

## 1. Feature Status Matrix

| Feature | Status | Implementation Details / Evidence |
|---|---|---|
| **Sources** | DONE | OddsAPI (Soccer/Basket). Norm schema in `src/ingestion`. |
| **Prediction** | DONE | CatBoost (Soccer). Baseline (Simple). |
| **Calibration** | DONE | Isotonic Regression. Evidence: `evidence/calibration_before_after.json`. |
| **EV Logic** | DONE | `(P_model * Odds) - 1`. Commission handling in Backtest. |
| **Guardrails** | DONE | Odds<=10.0, EV<=35%, SumProb≈1. Evidence: `evidence/outliers.csv`. |
| **Staking** | DONE | Fractional Kelly (0.5) with 5% Cap. Evidence: `evidence/top_ev_bets.csv` (stake_pct). |
| **Backtest** | DONE | Time-split OOS (2024+). Evidence: `evidence/backtest_summary.json`. |
| **Alerting** | DONE | Telegram Bot (Table format). Dedup 48h. Evidence: `evidence/telegram_preview.txt`. |
| **Scheduling** | DONE | 12:00/22:00 Vienna. `deploy/systemd/` & `evidence/schedule_test.log`. |
| **Basketball** | PARTIAL | Logic implemented, but Model skipped if missing. |

## 2. Repository Structure

See `evidence/repo_tree.txt` for full tree.
- `src/`: Core logic (strategy, models, integration).
- `scripts/`: Entrypoints (`run_value_finder.py`, `run_scheduler.py`).
- `deploy/`: Infrastructure configs.
- `audit_pack/`: Logs and Artifacts.

## 3. CLI & UX

Main Entrypoint: `scripts/run_value_finder.py`
Commands:
- `python scripts/run_value_finder.py --now --telegram --top 5` (Production)
- `python scripts/run_value_finder.py --dry-run` (Safe test)
- `python scripts/run_value_finder.py --help` (See `evidence/cli_help.txt`)

## 4. Models

Manifest: `evidence/models_manifest.json`
- **Soccer**: CatBoost Classifier (`catboost_v1_latest.pkl`).
- **Basketball**: Disabled/Conditional (file missing in this dump, logic handles it).

## 5. Metrics & Quality

- **LogLoss/ECE**: See `evidence/metrics_summary.json`.
- **Calibration**: Improved ECE by ~30% (See `evidence/calibration_before_after.json`).
- **Integrity**: 0 outliers in final sets (See `run_samples/top_ev_bets.csv`).

## 6. Live Outputs (Sample)

Latest snapshot in `run_samples/`:
- `predictions.csv`: 31 candidates.
- `top_ev_bets.csv`: 5 final bets.
- `alerts_sent.csv`: Log of sent telegram messages.

## 7. Blockers & Risks

See `BLOCKERS.md` for details.
- **Risk**: OddsAPI rate limits if schedule frequency increased.
- **Todo**: Retrain basketball model for v5.1.
