# FIX_INPUTS_PACKAGE: Evidence-Based Evaluation Report

**Date**: 2026-02-01  
**Purpose**: Evidence package for implementing next fixes safely

---

## Part 1: Data Integrity & Split Correctness

### 1.1 Dataset Sorting ✅

**Proof**: Dataset is globally sorted by `kickoff_time` before any split.

```python
# From: scripts/per_league_market_benchmark.py:168
df = df.sort_values('kickoff_time').reset_index(drop=True)
logger.info(f"Sorted: {df['kickoff_time'].is_monotonic_increasing}")
```

**Output**:
```
INFO:__main__:Sorted: True
```

### 1.2 Train/Valid/Test Date Ranges

| League | Train Start | Train End | Valid End | Test End | Train N | Valid N | Test N |
|--------|-------------|-----------|-----------|----------|---------|---------|--------|
| Championship | 2021-08-06 | 2023-09-02 | 2024-01-27 | 2024-05-11 | 1159 | 248 | 249 |
| Ligue 1 | 2021-08-06 | 2023-09-02 | 2024-01-27 | 2024-05-19 | 746 | 160 | 160 |
| La Liga | 2021-08-13 | 2023-09-02 | 2024-02-03 | 2024-05-26 | 798 | 171 | 171 |
| EPL | 2021-08-13 | 2023-09-02 | 2024-02-03 | 2024-05-19 | 798 | 171 | 171 |
| Bundesliga | 2021-08-13 | 2023-09-01 | 2024-02-03 | 2024-05-18 | 642 | 138 | 138 |
| Serie A | 2021-08-21 | 2023-09-02 | 2024-02-11 | 2024-06-02 | 797 | 171 | 171 |

### 1.3 Class Distribution on Test Set

| League | %Home | %Draw | %Away | Total |
|--------|-------|-------|-------|-------|
| Championship | 44% | 22% | 34% | 249 |
| Ligue 1 | 37% | 22% | 41% | 160 |
| La Liga | 43% | 27% | 30% | 171 |
| **EPL** | 45% | 23% | 32% | 171 |
| Bundesliga | 41% | 28% | 31% | 138 |
| Serie A | 36% | **33%** | 30% | 171 |

> [!NOTE]
> Serie A has highest draw rate (33%), making predictions harder.

### 1.4 No Leakage Confirmation ✅

**ELO/Features computed from past matches only**:

```python
# From: scripts/build_dataset_v2.py:290-310
# Pre-compute all ELO ratings chronologically
elo_snapshots: Dict[int, Tuple[float, float]] = {}

for idx, row in df.iterrows():
    home = str(row['HomeTeam'])
    away = str(row['AwayTeam'])
    
    # Save ELO BEFORE update (for this match's features)
    elo_snapshots[idx] = (elo.get_rating(home), elo.get_rating(away))
    
    # Update ELO AFTER for next matches
    result = get_result_from_score(fthg, ftag)
    elo.update(home, away, result)
```

**No post-match columns in feature set**:

```python
# From: models/feature_columns.json (28 features)
# NO columns like: FTHG, FTAG, FTR, result, score, goals
```

**Odds features = pre-match Pinnacle/median line**:

```python
# From: src/data/ml_odds_builder.py:196-220
# Uses bookmaker odds from event data (pre-match snapshot)
# Information time: odds at time of fetch (before kickoff)
```

### 1.5 Sample Rows (First 10)

```
         kickoff_time       league          home_team       away_team  odds_home  odds_draw  odds_away  elo_home  elo_away  label
  2021-08-06 00:00:00 championship          Blackburn    Bournemouth       3.94       3.45       2.00    1500.0    1500.0      2
  2021-08-06 00:00:00 championship        Bournemouth      West Brom       1.86       3.57       4.67    1500.0    1500.0      0
  2021-08-07 00:00:00 championship              Stoke        Reading       2.10       3.36       3.94    1500.0    1500.0      0
  2021-08-07 00:00:00 championship              Derby  Huddersfield       4.71       3.81       1.76    1500.0    1500.0      1
```

> [!NOTE]
> ELO starts at 1500 for all teams (cold-start expected in first matches).

---

## Part 2: Market Baseline vs Model

### 2.1 Summary Table

| League | #Test | %H/%D/%A | Mkt LL | Mdl LL | Δ LL | Mkt Brier | Mdl Brier | Mkt Acc | Mdl Acc |
|--------|-------|----------|--------|--------|------|-----------|-----------|---------|---------|
| Championship | 249 | 44/22/34 | 1.0268 | 1.0476 | **+0.021** | 0.2044 | 0.2095 | 49.4% | 49.4% |
| Ligue 1 | 160 | 37/22/41 | 1.0204 | 1.0297 | **+0.009** | 0.2045 | 0.2064 | 51.3% | 48.8% |
| La Liga | 171 | 43/27/30 | 0.9614 | 0.9677 | **+0.006** | 0.1908 | 0.1919 | 55.0% | 53.2% |
| EPL | 171 | 45/23/32 | 0.8762 | 0.9058 | **+0.030** | 0.1700 | 0.1759 | 62.0% | 60.8% |
| Bundesliga | 138 | 41/28/31 | 0.9440 | 0.9707 | **+0.027** | 0.1865 | 0.1932 | 56.5% | 55.8% |
| Serie A | 171 | 36/33/30 | 1.0142 | 1.0299 | **+0.016** | 0.2041 | 0.2073 | 49.1% | 46.2% |

> [!IMPORTANT]
> **Model is worse than market baseline in ALL leagues** (positive Δ LL = model worse).
> This is expected: the market already incorporates all available information.

### 2.2 Interpretation

- **EPL gap is largest** (+0.030 LL): most efficient market, hardest to beat
- **La Liga gap smallest** (+0.006 LL): potential edge exists
- **Model accuracy ≤ market accuracy** in all cases

### 2.3 Draw Calibration (EPL Example)

**Market Draw Calibration**:
| Bin | Mean Pred | Mean Actual | Count |
|-----|-----------|-------------|-------|
| 0.0-0.1 | 0.07 | 0.00 | 6 |
| 0.1-0.2 | 0.16 | 0.19 | 48 |
| 0.2-0.3 | 0.25 | **0.26** | 117 |

**Model Draw Calibration**:
| Bin | Mean Pred | Mean Actual | Count |
|-----|-----------|-------------|-------|
| 0.1-0.2 | 0.17 | 0.14 | 42 |
| 0.2-0.3 | 0.24 | **0.25** | 124 |
| 0.3-0.4 | 0.30 | **0.60** | 5 |

> [!WARNING]
> Model predicts 30% draw probability for 5 matches that were actually 60% draws.
> Small sample but indicates underconfidence in draws.

---

## Part 3: Time-Decay Gradient Experiment

### 3.1 Walk-Forward CV Setup

- **Fold structure**: Quarterly (90-day) validation windows
- **Folds per league**: 7 folds
- **Minimum training samples**: 200
- **Half-lives tested**: 30, 60, 90, 180, 365, 730, 1095, ∞ days

### 3.2 Fold Details (EPL Example)

| Fold | Train Period | Train N | Valid Period | Valid N |
|------|--------------|---------|--------------|---------|
| 0 | 2021-08-13 → 2022-08-07 | 390 | 2022-08-13 → 2022-11-06 | 126 |
| 1 | 2021-08-13 → 2022-11-06 | 516 | 2022-11-12 → 2023-02-08 | 74 |
| 2 | 2021-08-13 → 2023-02-08 | 590 | 2023-02-11 → 2023-05-08 | 137 |
| 3 | 2021-08-13 → 2023-05-08 | 727 | 2023-05-13 → 2023-05-28 | 33 |
| 4 | 2021-08-13 → 2023-05-28 | 760 | 2023-08-11 → 2023-11-05 | 109 |
| 5 | 2021-08-13 → 2023-11-05 | 869 | 2023-11-06 → 2024-02-03 | 114 |
| 6 | 2021-08-13 → 2024-02-03 | 983 | 2024-02-04 → 2024-05-03 | 125 |

### 3.3 Decay Gradient Results

#### EPL

| Half-life | LogLoss (mean±std) | Brier (mean±std) |
|-----------|-------------------|------------------|
| 30d | 0.9898±0.0313 | 0.1961±0.0077 |
| 60d | 0.9830±0.0303 | 0.1944±0.0073 |
| 90d | 0.9796±0.0343 | 0.1937±0.0086 |
| 180d | 0.9720±0.0343 | 0.1920±0.0085 |
| 365d | 0.9702±0.0370 | 0.1917±0.0090 |
| 730d | 0.9668±0.0391 | 0.1910±0.0097 |
| 1095d | 0.9670±0.0395 | 0.1911±0.0098 |
| **∞** | **0.9655±0.0386** | **0.1908±0.0095** |

> **Best for EPL**: ∞ (no decay) — all historical data valuable

#### Championship

| Half-life | LogLoss (mean±std) | Brier (mean±std) |
|-----------|-------------------|------------------|
| 30d | 1.0622±0.0212 | 0.2135±0.0051 |
| 90d | 1.0550±0.0250 | 0.2119±0.0060 |
| 365d | 1.0482±0.0243 | 0.2103±0.0059 |
| 730d | 1.0463±0.0214 | 0.2099±0.0052 |
| **1095d** | **1.0446±0.0212** | **0.2094±0.0052** |
| ∞ | 1.0477±0.0188 | 0.2102±0.0045 |

> **Best for Championship**: 1095d (~3 years)

### 3.4 Recommended Decay Per League

| League | Best Decay | LogLoss (mean±std) | Interpretation |
|--------|------------|-------------------|----------------|
| **EPL** | ∞ (none) | 0.9655±0.0386 | Stable patterns, all data valuable |
| **Bundesliga** | 1095d | 0.9775±0.0462 | 3-year patterns persist |
| **La Liga** | 730d | 0.9764±0.0376 | 2-year sweet spot |
| **Serie A** | ∞ (none) | 0.9796±0.0526 | Stable hierarchies |
| **Ligue 1** | 1095d | 0.9942±0.0417 | 3-year patterns |
| **Championship** | 1095d | 1.0446±0.0212 | High variance, longer memory helps |

> [!NOTE]
> **Previous finding (180d for EPL) was from single-split validation.**
> Walk-forward CV with 7 folds shows ∞ is actually better (more stable).

### 3.5 Decay Gradient Visualization (Conceptual)

```
Data Age Weight Schedule (based on 730d half-life):
├── Last 30d:   weight = 97%
├── Last 180d:  weight = 84%
├── Last 365d:  weight = 71%
├── Last 730d:  weight = 50%
├── Last 1095d: weight = 35%
└── Older:      weight = 25%
```

---

## Part 4: Reproducibility Commands

### Generate Market Benchmark

```bash
cd /Users/macuser/Documents/something/stavki_value_system

# Full benchmark (all leagues)
python3 scripts/per_league_market_benchmark.py

# Single league
python3 scripts/per_league_market_benchmark.py --league epl
```

**Output files**:
- `market_benchmark_results.csv`
- `data_integrity_results.csv`

### Generate Decay Gradient CSV

```bash
cd /Users/macuser/Documents/something/stavki_value_system

# Full evaluation (all leagues, ~5 min)
python3 scripts/decay_gradient_eval.py

# Single league
python3 scripts/decay_gradient_eval.py --league epl
```

**Output files**:
- `decay_gradient_results.csv`
- `decay_gradient_folds.csv`

---

## Key Findings Summary

| Finding | Evidence | Implication |
|---------|----------|-------------|
| Model worse than market | +0.006 to +0.030 LL gap | Need calibration or ensemble |
| EPL most efficient | Largest gap (+0.030 LL) | Hardest league to beat |
| Optimal decay varies by league | CV shows ∞/730d/1095d best | Use per-league config |
| Previous 180d finding was wrong | Single-split vs 7-fold CV | Walk-forward reveals ∞ better |
| Serie A high draw rate | 33% draws vs 22-27% others | Consider draw-specific modeling |
| Data integrity confirmed | Sorted, no leakage columns | Safe to proceed |
