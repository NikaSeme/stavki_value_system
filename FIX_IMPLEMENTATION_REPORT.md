# ML Pipeline Fix Implementation Report

> **Date**: 2026-02-01  
> **Tasks**: A (Outcome Mapping), B (ML Odds Line), C (Feature Contract), D (Retraining)

---

## Summary of Changes

| Task | Problem | Fix | Status |
|------|---------|-----|--------|
| **A** | Draw only recognized as "Draw", fake 3.5 fallback | Recognize Draw/Tie/X, skip events with missing outcomes | ✅ |
| **B** | Biased single_book selection (max home odds) | Pinnacle-first, median fallback, separate from bet execution | ✅ |
| **C** | 42 training vs ~28 live features, silent zeros | Strict 28-feature contract, hard fail on mismatch | ✅ |
| **D** | Model trained on corrupted pipeline | Retrained with V2 pipeline, chronological split | ✅ |

---

## Task A: Outcome Mapping Fix

### Code Changes

| File | Function | Change |
|------|----------|--------|
| [ml_odds_builder.py](file:///Users/macuser/Documents/something/stavki_value_system/src/data/ml_odds_builder.py#L16-18) | `DRAW_NAMES` | Added `{"draw", "tie", "x"}` constant |
| [ml_odds_builder.py](file:///Users/macuser/Documents/something/stavki_value_system/src/data/ml_odds_builder.py#L64-68) | `is_draw_outcome()` | Case-insensitive check against DRAW_NAMES |
| [ml_odds_builder.py](file:///Users/macuser/Documents/something/stavki_value_system/src/data/ml_odds_builder.py#L123-135) | `extract_hda_odds_from_bookmaker()` | Returns `None` if any H/D/A missing (no fallback) |

### Evidence from Audit

```
📋 Draw name recognition:
  'Draw' → ✅ DRAW
  'Tie' → ✅ DRAW
  'X' → ✅ DRAW
  
📋 Outcome classification:
  ✅ classify('Draw', ...) → D
  ✅ classify('Tie', ...) → D
  ✅ classify('X', ...) → D
```

### Unit Tests

27 tests all passing:

```
tests/test_ml_odds_builder.py::TestDrawRecognition::test_draw_lowercase PASSED
tests/test_ml_odds_builder.py::TestDrawRecognition::test_tie_lowercase PASSED
tests/test_ml_odds_builder.py::TestDrawRecognition::test_x_lowercase PASSED
tests/test_ml_odds_builder.py::TestExtractHDAOdds::test_missing_draw PASSED → Returns None
tests/test_ml_odds_builder.py::TestExtractHDAOdds::test_missing_home PASSED → Returns None
tests/test_ml_odds_builder.py::TestExtractHDAOdds::test_missing_away PASSED → Returns None
tests/test_ml_odds_builder.py::TestExtractHDAOdds::test_tie_as_draw PASSED
```

---

## Task B: ML Odds Line Builder

### Code Changes

| File | Function | Change |
|------|----------|--------|
| [ml_odds_builder.py](file:///Users/macuser/Documents/something/stavki_value_system/src/data/ml_odds_builder.py#L196-240) | `build_ml_odds_line()` | Pinnacle-first, median fallback, computes diagnostics |
| [build_dataset_v2.py](file:///Users/macuser/Documents/something/stavki_value_system/scripts/build_dataset_v2.py#L190-250) | `build_ml_line_from_historical_odds()` | Same logic for historical CSV data |

### Strategy

1. **Pinnacle first**: If bookmaker `"pinnacle"` in data, use their full line
2. **Median fallback**: Otherwise, take median of all books per outcome
3. **Diagnostics**: Compute `book_count`, `overround`, `dispersion_*`

### Evidence from Audit

```
📊 Sample ML odds lines:

  Newcastle United vs Aston Villa:
    Source: pinnacle
    Books available: 24
    ML Line: H=2.07, D=3.59, A=3.68
    Overround: 1.0334
    No-vig: H=0.467, D=0.270, A=0.263
    Dispersion: H=0.040, D=0.098, A=0.123

  📈 Line source distribution:
    Pinnacle: 5
    Median: 0
    Skipped: 0
```

### Dataset Build Log

```
Total matches: 7060
Processed: 7059
Skipped (no odds): 1
Pinnacle lines: 7058  ← 99.99% Pinnacle
Median lines: 1
```

---

## Task C: Strict Feature Contract

### Code Changes

| File | Purpose |
|------|---------|
| [feature_columns.json](file:///Users/macuser/Documents/something/stavki_value_system/models/feature_columns.json) | Canonical 28-feature list |
| [feature_contract.py](file:///Users/macuser/Documents/something/stavki_value_system/src/models/feature_contract.py) | Strict enforcement, hard fail on mismatch |

### Canonical Feature List (28)

```json
[
  "odds_home", "odds_draw", "odds_away",
  "implied_home", "implied_draw", "implied_away",
  "no_vig_home", "no_vig_draw", "no_vig_away",
  "market_overround",
  "line_dispersion_home", "line_dispersion_draw", "line_dispersion_away",
  "book_count",
  "elo_home", "elo_away", "elo_diff",
  "form_pts_home_l5", "form_pts_away_l5",
  "form_gf_home_l5", "form_gf_away_l5",
  "form_ga_home_l5", "form_ga_away_l5",
  "rest_days_home", "rest_days_away",
  "league", "home_team", "away_team"
]
```

### Evidence from Audit

```
📊 V2 Dataset Analysis:
  Samples: 7059
  Columns: 30 (28 features + label + kickoff_time)

  Feature alignment:
    Contract: 28
    Dataset:  28
    Common:   28
    Missing:  0 ✅ None
    Extra:    0 ✅ None

  ✅ PERFECT MATCH: Train features == Contract features
```

### V1 vs V2 Comparison

| Metric | V1 (Old) | V2 (New) |
|--------|----------|----------|
| Features | 42 | 28 |
| Book-specific odds | B365, PS, Max, Avg | Unified ML line |
| Sentiment/Injury | Included | Removed (unreliable) |
| Silent zero-fill | Yes | No (hard fail) |

---

## Task D: Model Retraining

### Configuration

```json
{
  "iterations": 1000,
  "depth": 5,
  "learning_rate": 0.03,
  "l2_leaf_reg": 9,
  "loss_function": "MultiClass",
  "early_stopping_rounds": 50
}
```

### Data Splits (Chronological)

| Split | Samples | Date Range |
|-------|---------|------------|
| Train | 4941 | 2021-08-06 → 2023-09-02 |
| Valid | 1059 | 2023-09-02 → 2024-01-20 |
| Test | 1059 | 2024-01-20 → 2024-06-02 |

### Metrics Comparison

| Metric | V1 (Old) | V2 (New) | Change |
|--------|----------|----------|--------|
| Features | 42 | 28 | -14 |
| Test Accuracy | 53.45% | 52.97% | -0.48% |
| Test LogLoss | 0.9788 | 0.9845 | +0.57% |
| Test Brier | N/A | 0.1957 | N/A |
| Calibration | Used | Not used (hurts) | - |

> **Note**: Slight accuracy drop expected when removing corrupt features (book-specific odds, sentiment). Model is now cleaner and more stable.

### Calibration Tables (Test Set)

**Home (selected bins)**:
| Bin | Pred | Actual | Count |
|-----|------|--------|-------|
| 0.3-0.4 | 0.349 | 0.375 | 176 |
| 0.6-0.7 | 0.653 | 0.642 | 120 |
| 0.7-0.8 | 0.740 | 0.696 | 125 |

### Artifacts Saved

- `models/catboost_v2.cbm`
- `models/catboost_v2_metadata.json`
- `models/feature_columns_v2.json`

---

## Full Audit Output

```
$ python3 scripts/audit_pipeline.py

======================================================================
🔍 ML PIPELINE AUDIT
======================================================================

AUDIT A: 1X2 OUTCOME MAPPING → ✅ PASS
AUDIT B: ML ODDS LINE CONSTRUCTION → ✅ PASS
AUDIT C: FEATURE CONTRACT ENFORCEMENT → ✅ PERFECT MATCH
AUDIT D: MODEL METRICS → ✅ PASS

======================================================================
✅ AUDIT COMPLETE
======================================================================
```

---

## Files Changed/Created

| File | Type |
|------|------|
| `src/data/ml_odds_builder.py` | **NEW** - ML odds line builder |
| `src/models/feature_contract.py` | **UPDATED** - Strict enforcement |
| `models/feature_columns.json` | **NEW** - Canonical feature list |
| `scripts/build_dataset_v2.py` | **NEW** - V2 dataset builder |
| `scripts/train_catboost_v2.py` | **NEW** - V2 training script |
| `scripts/audit_pipeline.py` | **NEW** - Audit checks |
| `tests/test_ml_odds_builder.py` | **NEW** - 27 unit tests |
| `data/ml_dataset_v2.csv` | **NEW** - V2 training data |
| `models/catboost_v2.cbm` | **NEW** - Trained model |
