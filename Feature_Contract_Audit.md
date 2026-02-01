# Feature Contract Audit

> **Date**: 2026-02-01  
> **Scope**: Evidence-based comparison of training vs live inference features (NO CODE CHANGES)

---

## 1. Training Feature Column List (CatBoost Model)

**Source**: [metadata_v1_20260131_201454.json](file:///Users/macuser/Documents/something/stavki_value_system/models/metadata_v1_20260131_201454.json)

```json
"features": [
    "OddsHome", "OddsDraw", "OddsAway",
    "B365H", "B365D", "B365A",
    "PSH", "PSD", "PSA",
    "MaxH", "MaxD", "MaxA",
    "AvgH", "AvgD", "AvgA",
    "HomeEloBefore", "AwayEloBefore",
    "EloExpHome", "EloExpAway", "EloDiff",
    "SentimentHome", "SentimentAway",
    "HomeInjury", "AwayInjury",
    "Market_Consensus", "Sharp_Divergence", "Odds_Volatility",
    "Home_GF_L5", "Home_GA_L5", "Home_Pts_L5",
    "Away_GF_L5", "Away_GA_L5", "Away_Pts_L5",
    "Home_Overall_GF_L5", "Home_Overall_GA_L5", "Home_Overall_Pts_L5",
    "Away_Overall_GF_L5", "Away_Overall_GA_L5", "Away_Overall_Pts_L5",
    "HomeTeam", "AwayTeam", "League"
]
```

**Total: 42 features**

---

## 2. Live Inference Feature Column List

**Source**: [live_extractor.py](file:///Users/macuser/Documents/something/stavki_value_system/src/features/live_extractor.py#L143-189)

```python
feat_dict = {
    'HomeEloBefore': home_elo,
    'AwayEloBefore': away_elo,
    'EloDiff': elo_diff,
    
    'Home_Pts_L5': home_form_raw['points_avg'],
    'Home_GF_L5': home_form_raw['goals_for_avg'],
    'Home_GA_L5': home_form_raw['goals_against_avg'],
    'Home_WinRate_L5': home_form_raw['win_rate'],
    
    'Away_Pts_L5': away_form_raw['points_avg'],
    'Away_GF_L5': away_form_raw['goals_for_avg'],
    'Away_GA_L5': away_form_raw['goals_against_avg'],
    
    'Home_Overall_Pts_L5': home_form_raw['points_avg'],
    'Home_Overall_GF_L5': home_form_raw['goals_for_avg'],
    'Home_Overall_GA_L5': home_form_raw['goals_against_avg'],
    
    'Away_Overall_Pts_L5': away_form_raw['points_avg'],
    'Away_Overall_GF_L5': away_form_raw['goals_for_avg'],
    'Away_Overall_GA_L5': away_form_raw['goals_against_avg'],
    
    'WinStreak_L5': ...,
    'LossStreak_L5': ...,
    'DaysSinceLastMatch': ...,

    'HomeTeam': home_team,
    'AwayTeam': away_team,
    'Season': '2023-24',
    'League': event.get('league', 'Unknown'),
    
    **market_feats,  # MarketProbHomeNoVig, MarketProbDrawNoVig, etc.
    **h2h_feats,     # H2HHomeWins, H2HDraws, etc.
}
```

**Live market_feats** (from `_extract_market_features`):
```
MarketProbHomeNoVig, MarketProbDrawNoVig, MarketProbAwayNoVig, OddsHomeAwayRatio
```

**Live h2h_feats** (from `_extract_h2h_features`):
```
H2HHomeWins, H2HDraws, H2HAwayWins, H2HHomeGoalsAvg, H2HAwayGoalsAvg
```

**Total (approximate): ~28 features**

---

## 3. Feature Comparison Table

| Category | Training Only | Live Only | Both |
|----------|---------------|-----------|------|
| **Odds** | OddsHome, OddsDraw, OddsAway | - | - |
| **Bet365** | B365H, B365D, B365A | - | - |
| **Pinnacle** | PSH, PSD, PSA | - | - |
| **Max** | MaxH, MaxD, MaxA | - | - |
| **Avg** | AvgH, AvgD, AvgA | - | - |
| **Elo** | EloExpHome, EloExpAway | - | HomeEloBefore, AwayEloBefore, EloDiff |
| **Form** | - | Home_WinRate_L5, WinStreak_L5, LossStreak_L5, DaysSinceLastMatch | Home_Pts_L5, Home_GF_L5, Home_GA_L5, Away_* |
| **Overall Form** | - | - | Home_Overall_*, Away_Overall_* |
| **Sentiment** | SentimentHome, SentimentAway | - | - |
| **Injury** | HomeInjury, AwayInjury | - | - |
| **Market Intel** | Market_Consensus, Sharp_Divergence, Odds_Volatility | MarketProbHomeNoVig, MarketProbDrawNoVig, MarketProbAwayNoVig, OddsHomeAwayRatio | - |
| **H2H** | - | H2HHomeWins, H2HDraws, H2HAwayWins, H2HHomeGoalsAvg, H2HAwayGoalsAvg | - |
| **Categorical** | - | Season | HomeTeam, AwayTeam, League |

### Summary

| Metric | Count |
|--------|-------|
| **Training features** | 42 |
| **Live features** | ~28 |
| **In both** | ~18 |
| **Training only** | ~24 |
| **Live only** | ~10 |
| **Match rate** | ~43% |

---

## 4. How Missing Features Are Filled at Inference

**Evidence** — from [live_extractor.py](file:///Users/macuser/Documents/something/stavki_value_system/src/features/live_extractor.py#L198-204):

```python
# Fill NA: Numeric with 0.0, Categorical with "Unknown"
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(exclude=[np.number]).columns

df[num_cols] = df[num_cols].fillna(0.0)
df[cat_cols] = df[cat_cols].fillna("Unknown")
```

**Evidence** — from [value_live.py](file:///Users/macuser/Documents/something/stavki_value_system/src/strategy/value_live.py#L403-407):

```python
# Add missing columns (fill with 0.0)
missing_cols = set(expected_cols) - set(X.columns)
if missing_cols:
    for col in missing_cols:
        X[col] = 0.0
```

### Fill Strategy

| Feature Type | Fill Value |
|--------------|------------|
| Numeric (missing) | `0.0` |
| Categorical (missing) | `"Unknown"` |
| NaN in existing numeric | `0.0` |
| NaN in existing categorical | `"Unknown"` |

---

## 5. Sample Live Feature Row

**Evidence** — from test output in [live_extractor.py](file:///Users/macuser/Documents/something/stavki_value_system/src/features/live_extractor.py#L509-540):

```python
features = extractor.extract_features(events, odds)

# Shape: (1, 22) 
# Columns: [
#   'HomeEloBefore', 'AwayEloBefore', 'EloDiff',
#   'Home_Pts_L5', 'Home_GF_L5', 'Home_GA_L5', 'Home_WinRate_L5',
#   'Away_Pts_L5', 'Away_GF_L5', 'Away_GA_L5',
#   'Home_Overall_Pts_L5', 'Home_Overall_GF_L5', 'Home_Overall_GA_L5',
#   'Away_Overall_Pts_L5', 'Away_Overall_GF_L5', 'Away_Overall_GA_L5',
#   'WinStreak_L5', 'LossStreak_L5', 'DaysSinceLastMatch',
#   'HomeTeam', 'AwayTeam', 'Season', 'League',
#   'MarketProbHomeNoVig', 'MarketProbDrawNoVig', 'MarketProbAwayNoVig',
#   'OddsHomeAwayRatio', 'H2HHomeWins', 'H2HDraws', 'H2HAwayWins',
#   'H2HHomeGoalsAvg', 'H2HAwayGoalsAvg', 'sharp_move_detected',
#   'odds_volatility', 'time_to_match_hours', 'market_efficiency_score'
# ]

# Sample values:
#   HomeEloBefore: 1500.0  (default)
#   AwayEloBefore: 1500.0  (default)
#   MarketProbHomeNoVig: 0.400 (calculated from odds)
```

---

## Critical Finding: Mismatch Impact

### Features Used in Training BUT NOT Available Live

| Feature | Impact |
|---------|--------|
| `OddsHome, OddsDraw, OddsAway` | Raw odds — live uses MarketProb* instead |
| `B365H, B365D, B365A` | Bet365-specific odds — not fetched |
| `PSH, PSD, PSA` | Pinnacle-specific odds — not fetched |
| `MaxH, MaxD, MaxA` | Max across books — not calculated |
| `AvgH, AvgD, AvgA` | Average across books — not calculated |
| `EloExpHome, EloExpAway` | Expected score features — not computed |
| `SentimentHome, SentimentAway` | Sentiment extracted separately, may not align |
| `HomeInjury, AwayInjury` | Injury data — not available live |
| `Market_Consensus, Sharp_Divergence` | Market intel features — different calculation |

### Features Available Live BUT NOT Used in Training

| Feature | Source |
|---------|--------|
| `Home_WinRate_L5` | Form calculation |
| `WinStreak_L5, LossStreak_L5` | Momentum features |
| `DaysSinceLastMatch` | Fatigue feature |
| `H2H*` | Head-to-head features |
| `MarketProbHomeNoVig` etc. | Market probability features |
| `Season` | Categorical (hardcoded) |

---

## Verdict

> [!CAUTION]
> **Only ~43% feature alignment between training and inference.**
> 
> The model was trained on 42 features including bookmaker-specific odds (B365, Pinnacle, Max, Avg) which are NOT available at inference time. The live extractor produces ~28 features, many of which were NOT used during training.
>
> Missing features are filled with `0.0`, which may cause:
> - Model receives out-of-distribution inputs
> - Predictions may be unreliable
> - Feature importance shifts unexpectedly
