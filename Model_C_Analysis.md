# Model C (Neural Network) — Analysis & Optimization Plan

**Date**: 2026-02-01  
**Status**: Currently DISABLED in production (`use_neural=False`)

---

## 1. Current State Summary

### 1.1 Architecture

| Component | Value |
|-----------|-------|
| **Class** | `DenseNN` (PyTorch) |
| **Layers** | Input → 64 → 32 → 16 → 3 |
| **Activation** | ReLU + BatchNorm + Dropout (0.3) |
| **Output** | 3-class softmax (Home/Draw/Away) |
| **Calibration** | Per-class Isotonic Regression |

### 1.2 Latest Training Metrics

**File**: [neural_metadata_20260131_195231.json](file:///Users/macuser/Documents/something/stavki_value_system/models/neural_metadata_20260131_195231.json)

| Metric | Neural (Model C) | CatBoost (Model B) | Delta |
|--------|------------------|--------------------| ------|
| **Accuracy** | 51.4% | 53.4% | **-2.0%** |
| **Log Loss** | 1.068 | 0.979 | **+0.089** |
| **Brier Score** | 0.201 | 0.194 | **+0.007** |

> [!CAUTION]
> **Model C performs significantly worse than CatBoost** on all metrics.
> This explains why `use_neural=False` is the default.

---

## 2. Critical Problems Identified

### 🔴 PROBLEM 1: Feature Dimension Mismatch

**Training Input Dim**: 35 features (from metadata)  
**Live Extractor Output**: 22 features  
**Ensemble Workaround**: Truncates to first 22 columns (lines 226-228 in `ensemble_predictor.py`)

```python
# Current hack in ensemble_predictor.py:205-228
if len(numeric_candidates.columns) > 22:
     logger.debug(f"Truncating Neural features from {len(numeric_candidates.columns)} to 22")
     numeric_candidates = numeric_candidates.iloc[:, :22]
```

**Impact**: Model receives wrong features → predictions are garbage.

---

### 🔴 PROBLEM 2: Model is Disabled by Default

**File**: [ensemble_predictor.py:33](file:///Users/macuser/Documents/something/stavki_value_system/src/models/ensemble_predictor.py#L33)

```python
def __init__(self, ensemble_file='...', use_neural=False):  # ← OFF by default
```

Only one caller enables it:
```python
# value_live.py:337
_ensemble = EnsemblePredictor(use_neural=True)
```

But even when enabled, the feature mismatch causes silent failures.

---

### 🔴 PROBLEM 3: No Feature Contract Enforcement

**Training** ([train_neural_model.py:316-324](file:///Users/macuser/Documents/something/stavki_value_system/scripts/train_neural_model.py#L316-324)):
- Dynamically selects all numeric columns
- Drops constant columns
- Feature list not saved explicitly

**Inference** ([neural_predictor.py:107-118](file:///Users/macuser/Documents/something/stavki_value_system/src/models/neural_predictor.py#L107-118)):
- Relies on `scaler.feature_names_in_` (may not exist)
- Falls back to "select numeric only"

**Impact**: No guarantee train/inference features match.

---

### 🟡 PROBLEM 4: Suboptimal Architecture

| Issue | Current | Best Practice |
|-------|---------|---------------|
| No skip connections | ✗ | ResNet-style residuals improve gradient flow |
| Fixed hidden dims | [64,32,16] | Should be tuned per dataset |
| No learning rate scheduler | ✗ | ReduceLROnPlateau improves convergence |
| No class weights | ✗ | Draws are underrepresented (~25%) |
| Single point calibration | val set only | Platt scaling on held-out test |

---

### 🟡 PROBLEM 5: No League-Specific Models

CatBoost has per-league models (`soccer_epl.cbm`, `soccer_bundesliga.cbm`, etc.).  
Neural uses a **single global model** trained on all leagues.

**Impact**: Neural can't capture league-specific patterns.

---

### 🟡 PROBLEM 6: League Weights Ignore Neural Quality

**File**: [league_weights.json](file:///Users/macuser/Documents/something/stavki_value_system/models/league_weights.json)

```json
"soccer_epl": {
  "weights": {
    "catboost": 0.457,
    "neural": 0.23,   // ← Still 23% weight despite being broken!
    "poisson": 0.313
  }
}
```

Weights were optimized assuming Neural works correctly.

---

## 3. Missing APIs & Integration Points

### 3.1 Required Methods in `NeuralPredictor`

| Method | Status | Needed For |
|--------|--------|------------|
| `get_feature_names()` | ❌ Missing | Feature contract validation |
| `predict_with_uncertainty()` | ❌ Missing | Confidence-weighted ensembling |
| `per_league_predict()` | ❌ Missing | League-specific routing |
| `get_calibration_stats()` | ❌ Missing | Calibration quality monitoring |

### 3.2 Required Training Script Enhancements

| Feature | Status | Needed For |
|---------|--------|------------|
| Save `feature_columns.json` | ❌ Missing | Reproducible inference |
| Per-league training loop | ❌ Missing | League-specific models |
| Hyperparameter tuning (Optuna) | ❌ Missing | Architecture optimization |
| Cross-validation | ❌ Missing | Robust metric estimation |
| Learning rate scheduler | ❌ Missing | Faster convergence |
| Class weighting | ❌ Missing | Draw prediction improvement |

### 3.3 Required Ensemble Integration

| Change | Status | Impact |
|--------|--------|--------|
| Feature alignment check | ❌ Missing | Prevent garbage predictions |
| Uncertainty-based weighting | ❌ Missing | Weight models by confidence |
| Dynamic enable/disable | ❌ Missing | Skip Neural if metrics bad |

---

## 4. Optimization Roadmap

### Phase 1: Fix Feature Contract (Critical)

1. **Save feature columns during training**:
   ```python
   # In train_neural_model.py
   with open('models/neural_feature_columns.json', 'w') as f:
       json.dump(feature_cols, f)
   ```

2. **Load and validate in NeuralPredictor**:
   ```python
   def get_feature_names(self):
       return self.feature_columns  # Loaded from JSON
   ```

3. **Update LiveFeatureExtractor** to produce matching features.

---

### Phase 2: Architecture Improvements

```python
class ImprovedDenseNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.3):
        super().__init__()
        
        # Residual blocks
        self.block1 = ResidualBlock(input_dim, hidden_dims[0], dropout)
        self.block2 = ResidualBlock(hidden_dims[0], hidden_dims[1], dropout)
        self.block3 = ResidualBlock(hidden_dims[1], hidden_dims[2], dropout)
        
        # Output
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dims[2]),  # LayerNorm instead of BatchNorm
            nn.Linear(hidden_dims[2], 3)
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.output(x)
```

---

### Phase 3: Per-League Training

```python
for league in ['epl', 'bundesliga', 'laliga', 'seriea', 'ligue1', 'championship']:
    df_league = df[df['League'] == league]
    train_neural_for_league(df_league, output=f'models/neural_{league}.pt')
```

---

### Phase 4: Hyperparameter Optimization

```python
import optuna

def objective(trial):
    hidden_dim = trial.suggest_int('hidden_dim', 32, 256)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    
    model = DenseNN(hidden_dims=[hidden_dim, hidden_dim//2, hidden_dim//4], dropout=dropout)
    # ... train and return validation loss
```

---

### Phase 5: Calibration Upgrade

Replace Isotonic with **Temperature Scaling** (more stable):

```python
class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits):
        return logits / self.temperature
```

---

## 5. Immediate Action Items

| Priority | Action | Effort | Impact |
|----------|--------|--------|--------|
| 🔴 P0 | Fix feature contract (save/load columns) | 2h | High |
| 🔴 P0 | Align LiveExtractor with training features | 4h | High |
| 🟡 P1 | Add learning rate scheduler | 1h | Medium |
| 🟡 P1 | Add class weights for draws | 1h | Medium |
| 🟡 P1 | Update league_weights.json (set neural=0 until fixed) | 30m | Medium |
| 🟢 P2 | Per-league training script | 8h | High |
| 🟢 P2 | Optuna hyperparameter search | 4h | Medium |
| 🟢 P2 | Residual block architecture | 4h | Medium |

---

## 6. Recommended Configuration (Until Fixed)

**Set Neural weight to 0 in production**:

```json
// models/league_weights.json
"soccer_epl": {
  "weights": {
    "catboost": 0.70,
    "neural": 0.0,    // ← DISABLE until fixed
    "poisson": 0.30
  }
}
```

**Or disable in code**:
```python
# value_live.py:337
_ensemble = EnsemblePredictor(use_neural=False)  # ← Explicit disable
```

---

## 7. File References

| Purpose | Path |
|---------|------|
| Predictor Class | [src/models/neural_predictor.py](file:///Users/macuser/Documents/something/stavki_value_system/src/models/neural_predictor.py) |
| Training Script | [scripts/train_neural_model.py](file:///Users/macuser/Documents/something/stavki_value_system/scripts/train_neural_model.py) |
| Ensemble Integration | [src/models/ensemble_predictor.py](file:///Users/macuser/Documents/something/stavki_value_system/src/models/ensemble_predictor.py) |
| Feature Extractor | [src/features/live_extractor.py](file:///Users/macuser/Documents/something/stavki_value_system/src/features/live_extractor.py) |
| League Weights | [models/league_weights.json](file:///Users/macuser/Documents/something/stavki_value_system/models/league_weights.json) |
| Latest Model | [models/neural_v1_latest.pt](file:///Users/macuser/Documents/something/stavki_value_system/models/neural_v1_latest.pt) |
| Latest Metadata | [models/neural_metadata_20260131_195231.json](file:///Users/macuser/Documents/something/stavki_value_system/models/neural_metadata_20260131_195231.json) |

---

**Conclusion**: Model C is currently broken due to feature mismatch and underperforming due to suboptimal architecture. The immediate fix is to disable it in production (set weight to 0). Full rehabilitation requires fixing the feature contract, improving architecture, and potentially training per-league models.
