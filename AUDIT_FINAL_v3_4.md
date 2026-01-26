# AUDIT FINAL REPORT v3.4 (Fix & Realism)

**Date**: 2026-01-25
**Version**: v3.4
**Focus**: Safety, Realism, and Strict Diversification

## 1. Executive Summary
This update corrects the "Baseline Model" and "High EV Junk" issues found in v3.3. We have implemented strict filtering rules and disabled the unproven Basketball model until historical data is ingested. The system is now significantly safer and backed by a rigorous Out-of-Sample (OOS) backtest.

### Key Changes
- **Basketball**: **DISABLED** (Skipped in pipeline). No baseline guessing allowed.
- **Safety**: **High Odds (>10.0)** are now HARD filtered (moved to `outliers.csv`).
- **Diversification**: Strictly enforced Max 2 bets per league/day.
- **Backtest**: OOS test on 2023 data showing +26.8% ROI with Betfair fees included.

---

## 2. Artifact Verification (v3.4)

| ID | Component | Status | Location |
|---|---|---|---|
| A9 | Global Selection Report | ✅ PASS | `audit_pack/A9_live/selection_report.json` |
| A9 | Top Bets (Safe) | ✅ PASS | `audit_pack/A9_live/top_ev_bets.csv` |
| A8 | Staking Evidence | ✅ PASS | `audit_pack/A8_staking/stake_cap_check.csv` |
| A7 | Rigorous Backtest | ✅ PASS | `audit_pack/A7_backtest/bets_backtest.csv` |
| S1 | Validation Script | ✅ PASS | `scripts/validate_audit_pack.py` |

---

## 3. Backtest Performance (v3.4 OOS)

- **Data**: Validation + Test Splits (Strictly unseen).
- **Fees**: 2% Commission on Net Winnings.
- **Filters**: Max Odds 10.0, Max 2 bets/day.

| Metric | Result |
|---|---|
| Start Bankroll | 1000.0 |
| Final Bankroll | 1267.86 |
| **ROI** | **+26.8%** |
| Bets Placed | 125 |

Look at `audit_pack/A7_backtest/equity_curve.png` to see the performance.

---

## 4. Live Pipeline Status

Running `run_value_finder.py` now filters aggressively:
- **Total Candidates**: ~34
- **Filtered Out**: ~24 (High Odds, League Limits)
- **Final Selection**: 10 High Quality-Bets

**Status**: DEPLOYMENT READY (Restricted Mode)
