"""
Feature Registry
Central source of truth for all feature column names.
Prevents mismatches between Engineering, Training, and Inference.
"""

# Base Features (Raw)
BASE_COLS = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'League', 'Season']

# Derived Features (Numerical)
NUMERIC_FEATURES = [
    # Home Specific Form
    'Home_GF_L5', 'Home_GA_L5', 'Home_Pts_L5',
    # Away Specific Form
    'Away_GF_L5', 'Away_GA_L5', 'Away_Pts_L5',
    # Overall Form (Phase 2 New!)
    'Home_Overall_GF_L5', 'Home_Overall_GA_L5', 'Home_Overall_Pts_L5',
    'Away_Overall_GF_L5', 'Away_Overall_GA_L5', 'Away_Overall_Pts_L5',
    # Head to Head (Phase 2 Future)
    # 'H2H_Home_Wins', 'H2H_Draws'
]

# Categorical Features
CATEGORICAL_FEATURES = [
    'HomeTeam',
    'AwayTeam',
    'League'
]

# Market Data
ODDS_FEATURES = [
    'OddsHome', 'OddsDraw', 'OddsAway'
]

# Columns to Exclude from Training (Leakage/Meta)
EXCLUDE_COLS = BASE_COLS + [
    'GoalDiff', 'TotalGoals', 'index', 'Unnamed: 0'
]

def get_feature_list(include_odds=True):
    """Get list of active features for model training."""
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    if include_odds:
        features += ODDS_FEATURES
    return features
