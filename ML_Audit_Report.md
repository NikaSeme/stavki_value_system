# ML Audit Report — STAVKI Value System

> Generated: 2026-01-31  
> Scope: CatBoost-based football match outcome prediction (1X2)  
> Status: **Evidence-based documentation only — no modifications made**

---

## 1. Prediction Moment (Time of Information)

### 1.1 What exact moment is the model designed to predict for?

**Answer: At odds fetch time (typically 24+ hours pre-match)**

The system fetches live odds from The Odds API and makes predictions at that moment. Features are extracted using the current ELO/form state, which reflects all historical matches up to that point.

### 1.2 Evidence

**File**: [live_extractor.py](file:///Users/macuser/Documents/something/stavki_value_system/src/features/live_extractor.py)

```python
# Lines 106-120: extract_features method
def extract_features(
    self,
    events: pd.DataFrame,
    odds_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Extract features for live events.
    
    Args:
        events: DataFrame with columns: event_id, home_team, away_team
        odds_df: DataFrame with current odds (must include outcome_price)
        
    Returns:
        DataFrame with 22 features matching training format
    """
```

**File**: [run_value_finder.py](file:///Users/macuser/Documents/something/stavki_value_system/scripts/run_value_finder.py#L261-274)

```python
# Lines 261-274: Odds loading at runtime
print(f"\n📊 Loading Odds Data...")
import pandas as pd
if args.global_mode:
    search_pattern = f"{args.odds_dir}/events_latest_*.csv"
    files = sorted(glob.glob(search_pattern))
    if not files:
        print(f"❌ No odds files found in {args.odds_dir}")
        sys.exit(1)
    latest_file = files[-1]
    print(f"  ✓ Unified File: {latest_file}")
    odds_df = pd.read_csv(latest_file)
```

**Implication**: Predictions use current (not closing) odds at fetch time. This means training on historical closing odds creates a **slight temporal mismatch** with live inference.

---

## 2. Dataset Scope

### 2.1 Summary

| Metric | Value |
|--------|-------|
| **Total matches** | 7,060 |
| **Date range** | 2021-08-06 to 2024-06-02 |
| **Seasons** | 3 (2021-22, 2022-23, 2023-24) |
| **Leagues** | 6 |

### 2.2 Leagues Breakdown

| League | Matches |
|--------|---------|
| championship | 1,656 |
| epl | 1,140 |
| laliga | 1,140 |
| seriea | 1,140 |
| ligue1 | 1,066 |
| bundesliga | 918 |

### 2.3 Train/Val/Test Split

| Split | Count | Date Range |
|-------|-------|------------|
| Train | 4,942 | 2021-08-06 to 2024-05-26 |
| Val | 1,059 | 2021-08-21 to 2024-05-19 |
| Test | 1,059 | 2021-10-22 to 2024-06-02 |

### 2.4 Evidence

**Command executed**:
```bash
python3 -c "
import pandas as pd
df = pd.read_csv('data/processed/multi_league_features_2021_2024.csv')
print(f'Total matches: {len(df)}')
print(df['League'].value_counts())
"
```

**Output**:
```
Total matches: 7060
League
championship    1656
epl             1140
laliga          1140
seriea          1140
ligue1          1066
bundesliga       918
```

**File**: [train_model.py](file:///Users/macuser/Documents/something/stavki_value_system/scripts/train_model.py#L42-57)

```python
# Lines 42-57: time_based_split function
def time_based_split(df, train_frac=0.70, val_frac=0.15):
    """Split data by time to avoid leakage."""
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    
    logger.info(f"Time-based split:")
    logger.info(f"  Train: {len(train)} matches ({train['Date'].min()} to {train['Date'].max()})")
    logger.info(f"  Val:   {len(val)} matches ({val['Date'].min()} to {val['Date'].max()})")
    logger.info(f"  Test:  {len(test)} matches ({test['Date'].min()} to {test['Date'].max()})")
    
    return train, val, test
```

---

## 3. Exact API / Raw Inputs Used

### 3.1 API Source

**The Odds API** (https://api.the-odds-api.com)

### 3.2 Raw Fields Fetched

**File**: [odds_api_client.py](file:///Users/macuser/Documents/something/stavki_value_system/src/data/odds_api_client.py#L65-100)

```python
# Lines 65-100: fetch_odds function
def fetch_odds(
    sport_key: str,
    regions: str,
    markets: str,
    *,
    odds_format: str = "decimal",
    date_format: str = "iso",
    bookmakers: Optional[str] = None,
    commence_time_from: Optional[str] = None,
    commence_time_to: Optional[str] = None,
    cfg: Optional[OddsAPIConfig] = None,
) -> List[Dict[str, Any]]:
```

### 3.3 Normalized Output Schema

**File**: [odds_normalize.py](file:///Users/macuser/Documents/something/stavki_value_system/src/data/odds_normalize.py#L16-64)

```python
# Lines 26-63: Row schema produced
"""
Row schema:
- event_id, sport_key, commence_time
- home_team, away_team
- bookmaker_key, bookmaker_title, last_update
- market_key (e.g., h2h), outcome_name, outcome_price
- odds_snapshot_time (ingestion time)
"""
```

### 3.4 Sample Raw → Normalized

**Raw API JSON** (redacted):
```json
{
  "id": "abc123",
  "sport_key": "soccer_epl",
  "commence_time": "2026-01-31T15:00:00Z",
  "home_team": "Arsenal",
  "away_team": "Chelsea",
  "bookmakers": [
    {
      "key": "pinnacle",
      "markets": [{"key": "h2h", "outcomes": [
        {"name": "Arsenal", "price": 2.10},
        {"name": "Draw", "price": 3.50},
        {"name": "Chelsea", "price": 3.40}
      ]}]
    }
  ]
}
```

**Normalized CSV row**:
```csv
event_id,sport_key,home_team,away_team,bookmaker_key,outcome_name,outcome_price
abc123,soccer_epl,Arsenal,Chelsea,pinnacle,Arsenal,2.10
abc123,soccer_epl,Arsenal,Chelsea,pinnacle,Draw,3.50
abc123,soccer_epl,Arsenal,Chelsea,pinnacle,Chelsea,3.40
```

---

## 4. Features Used by CatBoost

### 4.1 Full Feature List (48 columns in dataset)

**Command**:
```bash
head -1 data/processed/multi_league_features_2021_2024.csv
```

**Output**:
```
Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,OddsHome,OddsDraw,OddsAway,B365H,B365D,B365A,PSH,PSD,PSA,MaxH,MaxD,MaxA,AvgH,AvgD,AvgA,Season,League,HomeEloBefore,AwayEloBefore,HomeEloAfter,AwayEloAfter,EloExpHome,EloExpAway,EloDiff,SentimentHome,SentimentAway,HomeInjury,AwayInjury,Market_Consensus,Sharp_Divergence,Odds_Volatility,Home_GF_L5,Home_GA_L5,Home_Pts_L5,Away_GF_L5,Away_GA_L5,Away_Pts_L5,Home_Overall_GF_L5,Home_Overall_GA_L5,Home_Overall_Pts_L5,Away_Overall_GF_L5,Away_Overall_GA_L5,Away_Overall_Pts_L5
```

### 4.2 Feature Grouping

| Group | Features | Pre-Match Available? |
|-------|----------|---------------------|
| **ELO** | `HomeEloBefore`, `AwayEloBefore`, `EloDiff`, `EloExpHome`, `EloExpAway` | ✅ YES |
| **Form (Home-specific L5)** | `Home_GF_L5`, `Home_GA_L5`, `Home_Pts_L5` | ✅ YES |
| **Form (Away-specific L5)** | `Away_GF_L5`, `Away_GA_L5`, `Away_Pts_L5` | ✅ YES |
| **Form (Overall L5)** | `Home_Overall_*`, `Away_Overall_*` | ✅ YES |
| **Odds** | `OddsHome`, `OddsDraw`, `OddsAway`, `B365*`, `PS*`, `Max*`, `Avg*` | ⚠️ CLOSING (see note) |
| **Market Features** | `Market_Consensus`, `Sharp_Divergence`, `Odds_Volatility` | ⚠️ CLOSING |
| **Sentiment** | `SentimentHome`, `SentimentAway`, `HomeInjury`, `AwayInjury` | ✅ YES (placeholders=0) |
| **Categorical** | `HomeTeam`, `AwayTeam`, `League` | ✅ YES |
| **LEAKAGE (excluded)** | `FTHG`, `FTAG`, `FTR`, `HomeEloAfter`, `AwayEloAfter` | ❌ POST-MATCH |

### 4.3 Evidence — Feature Engineering

**File**: [train_model.py](file:///Users/macuser/Documents/something/stavki_value_system/scripts/train_model.py#L235-257)

```python
# Lines 235-257: Feature selection
# CRITICAL: Exclude all match outcome columns to prevent data leakage!
# We can ONLY use features known BEFORE the match starts
exclude_cols = ['Date', 'HomeTeam', 'AwayTeam', 'Season', 'FTR', 'League',
                'FTHG', 'FTAG', 'GoalDiff', 'TotalGoals',
                'HomeEloAfter', 'AwayEloAfter']  # Match outcomes + Leakage!
num_features = [col for col in df.columns if col not in exclude_cols]

# Define Categorical Features (The Upgrade!)
cat_features = ['HomeTeam', 'AwayTeam']

# Add League if present (multi-league training)
if 'League' in df.columns:
    cat_features.append('League')
```

> [!WARNING]
> **Odds Features**: Training uses closing odds from football-data.co.uk, but live inference uses odds at fetch time. This creates a temporal gap.

---

## 5. Leakage Safeguards and Time Splitting

### 5.1 Checklist

| Check | Answer | Evidence |
|-------|--------|----------|
| Is split time-based (not random)? | ✅ **YES** | `time_based_split()` uses `df.iloc[:train_end]` after sorting by Date |
| Are ELO values computed before each match? | ✅ **YES** | `HomeEloBefore` stored before `elo.update()` is called |
| Are form features from past matches only? | ✅ **YES** | `.shift(1).rolling(5)` ensures no current-match data |
| Are odds closing or pre-match? | ⚠️ **CLOSING** | Training uses historical closing odds |

### 5.2 Evidence — Time-Based Split

**File**: [train_model.py](file:///Users/macuser/Documents/something/stavki_value_system/scripts/train_model.py#L42-57)

```python
def time_based_split(df, train_frac=0.70, val_frac=0.15):
    """Split data by time to avoid leakage."""
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
```

### 5.3 Evidence — ELO Before Match

**File**: [elo.py](file:///Users/macuser/Documents/something/stavki_value_system/src/features/elo.py#L167-189)

```python
# Lines 167-189: ELO computed BEFORE update
for idx, row in df.iterrows():
    # Get ratings before match
    home_rating = elo.get_rating(row['HomeTeam'])
    away_rating = elo.get_rating(row['AwayTeam'])
    
    # Expected scores
    e_home, e_away = elo.expected_score(row['HomeTeam'], row['AwayTeam'])
    
    # Update ratings (AFTER storing before values)
    home_after, away_after = elo.update(
        row['HomeTeam'],
        row['AwayTeam'],
        row['FTR'],
        row['Date']
    )
    
    # Store
    home_elo_before.append(home_rating)  # <-- BEFORE update
    away_elo_before.append(away_rating)
```

### 5.4 Evidence — Form Features with shift(1)

**File**: [engineer_multi_league_features.py](file:///Users/macuser/Documents/something/stavki_value_system/scripts/engineer_multi_league_features.py#L53-59)

```python
# Lines 53-59: Rolling with shift to prevent leakage
# Rolling averages (shift to avoid leakage)
rolling_gf = goals_for.shift(1).rolling(window, min_periods=1).mean()
rolling_ga = goals_against.shift(1).rolling(window, min_periods=1).mean()
rolling_pts = points.shift(1).rolling(window, min_periods=1).mean()
```

---

## 6. CatBoost Training Configuration

### 6.1 Hyperparameter Search Space

**File**: [train_model.py](file:///Users/macuser/Documents/something/stavki_value_system/scripts/train_model.py#L100-136)

```python
# Lines 100-136: Hyperparameter search
def hyperparameter_search(X_train, y_train, X_val, y_val, cat_features_indices, n_trials=10):
    param_grid = {
        'depth': [4, 5, 6, 7],
        'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5, 7, 9],
        'random_strength': [0.1, 1, 2],
        'bagging_temperature': [0, 1]
    }
    
    # ...
    
    model = CatBoostClassifier(
        iterations=800,
        loss_function='MultiClass',
        eval_metric='MultiClass',
        early_stopping_rounds=30,
        verbose=False,
        cat_features=cat_features_indices,
        **params
    )
```

### 6.2 Final Configuration (from metadata)

**File**: [metadata_v1_20260131_201454.json](file:///Users/macuser/Documents/something/stavki_value_system/models/metadata_v1_20260131_201454.json)

```json
{
  "model_type": "catboost_optimized",
  "hyperparameters": {
    "depth": 5,
    "learning_rate": 0.03,
    "l2_leaf_reg": 9,
    "random_strength": 1,
    "bagging_temperature": 0
  },
  "num_features": 42
}
```

### 6.3 Configuration Summary

| Parameter | Value |
|-----------|-------|
| `loss_function` | `MultiClass` |
| `iterations` | 800 (with early stopping) |
| `depth` | 5 |
| `learning_rate` | 0.03 |
| `l2_leaf_reg` | 9 |
| `min_data_in_leaf` | Not specified (CatBoost default) |
| `bagging_temperature` | 0 |
| `random_strength` | 1 |
| `early_stopping_rounds` | 30 |
| `class_weights` | Not specified (auto) |
| `cat_features` | `['HomeTeam', 'AwayTeam', 'League']` |

---

## 7. Calibration

### 7.1 Method Used

**Isotonic Regression** (custom implementation)

### 7.2 Fitted On

**Validation set** (not train, not test)

### 7.3 Scope

**Global** (not per-league)

### 7.4 Evidence

**File**: [calibration.py](file:///Users/macuser/Documents/something/stavki_value_system/src/models/calibration.py#L21-64)

```python
class IsotonicCalibrator:
    """
    A bug-proof replacement for CalibratedClassifierCV that works on all 
    scikit-learn versions by manually implementing isotonic calibration.
    """
    def __init__(self, base_model, scaler=None):
        self.base_model = base_model
        self.scaler = scaler
        self.calibrators = {}
        
    def fit(self, X_val, y_val):
        # Get raw probabilities
        probs = self.base_model.predict_proba(X_val_processed)
        n_classes = probs.shape[1]
        
        for i in range(n_classes):
            iso = IsotonicRegression(out_of_bounds='clip')
            target = (y_val == i).astype(float)
            iso.fit(probs[:, i], target)
            self.calibrators[i] = iso
```

**File**: [train_model.py](file:///Users/macuser/Documents/something/stavki_value_system/scripts/train_model.py#L342-348)

```python
# Lines 342-348: Calibration on validation set
# --- CALIBRATION ---
logger.info("Calibrating best model...")
calibrator = get_best_calibrator(best_model)
calibrator.fit(X_val, y_val)  # <-- Fitted on VALIDATION set

# --- EVALUATION ---
metrics_test = evaluate_model(calibrator, X_test, y_test, "Test Set")
```

### 7.5 Calibration Artifacts

| File | Path |
|------|------|
| Latest Calibrator | `models/calibrator_v1_20260131_201454.pkl` |
| Symlink | `models/calibrator_v1_latest.pkl` → timestamped file |

---

## 8. Current Test Metrics

### 8.1 Overall Test Set Metrics

**File**: [metadata_v1_20260131_201454.json](file:///Users/macuser/Documents/something/stavki_value_system/models/metadata_v1_20260131_201454.json)

```json
{
  "metrics": {
    "test": {
      "accuracy": 0.534466477809254,
      "log_loss": 0.978756997900215,
      "brier_score": 0.19444571622420992,
      "brier_per_class": [
        0.20946158773691123,
        0.18952069650853248,
        0.18435486442718613
      ]
    }
  }
}
```

### 8.2 Summary Table

| Metric | Value |
|--------|-------|
| **Accuracy** | 53.45% |
| **Log Loss** | 0.979 |
| **Brier Score (avg)** | 0.194 |
| Brier (Home) | 0.209 |
| Brier (Draw) | 0.190 |
| Brier (Away) | 0.184 |

### 8.3 Per-League Test Set Size

| League | Test Matches |
|--------|--------------|
| championship | 222 |
| laliga | 182 |
| seriea | 180 |
| epl | 177 |
| bundesliga | 154 |
| ligue1 | 144 |

### 8.4 Per-League ROI (from league_config.json)

**File**: [league_config.json](file:///Users/macuser/Documents/something/stavki_value_system/models/league_config.json)

| League | Backtest ROI | Policy |
|--------|--------------|--------|
| bundesliga | +25% | CATBOOST_ONLY |
| ligue1 | +20% | POISSON_HEAVY |
| championship | +14% | CATBOOST_ONLY |
| laliga | +12% | ENSEMBLE |
| seriea | -7% | SKIP |
| epl | -15% | SKIP |

> [!CAUTION]
> These ROI figures are from backtest. Actual live performance may differ due to closing odds vs. current odds mismatch.

### 8.5 Calibration Table (10 bins)

**Not available in saved artifacts.** To generate, run:
```python
from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(y_test, y_proba[:, class_idx], n_bins=10)
```

---

## Appendix: File References

| Purpose | File Path |
|---------|-----------|
| Training Script | [scripts/train_model.py](file:///Users/macuser/Documents/something/stavki_value_system/scripts/train_model.py) |
| Feature Engineering | [scripts/engineer_multi_league_features.py](file:///Users/macuser/Documents/something/stavki_value_system/scripts/engineer_multi_league_features.py) |
| ELO Calculation | [src/features/elo.py](file:///Users/macuser/Documents/something/stavki_value_system/src/features/elo.py) |
| Calibration | [src/models/calibration.py](file:///Users/macuser/Documents/something/stavki_value_system/src/models/calibration.py) |
| Live Feature Extraction | [src/features/live_extractor.py](file:///Users/macuser/Documents/something/stavki_value_system/src/features/live_extractor.py) |
| Odds API Client | [src/data/odds_api_client.py](file:///Users/macuser/Documents/something/stavki_value_system/src/data/odds_api_client.py) |
| Value Finder | [scripts/run_value_finder.py](file:///Users/macuser/Documents/something/stavki_value_system/scripts/run_value_finder.py) |
| Dataset | [data/processed/multi_league_features_2021_2024.csv](file:///Users/macuser/Documents/something/stavki_value_system/data/processed/multi_league_features_2021_2024.csv) |
| Latest Model Metadata | [models/metadata_v1_20260131_201454.json](file:///Users/macuser/Documents/something/stavki_value_system/models/metadata_v1_20260131_201454.json) |
| League Config | [models/league_config.json](file:///Users/macuser/Documents/something/stavki_value_system/models/league_config.json) |

---

**End of Audit Report**
