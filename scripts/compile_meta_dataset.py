"""
Compile meta-dataset for meta-model training.

Collects predictions from all 3 models (Poisson, CatBoost, Neural)
and creates training data for the meta-learner.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_poisson_model import PoissonMatchPredictor
from scripts.train_neural_model import DenseNN, NeuralModelTrainer
from src.models.loader import ModelLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data():
    """Load features dataset."""
    base_dir = Path(__file__).parent.parent
    
    # Load features dataset (has everything we need)
    features_df = pd.read_csv(base_dir / 'data/processed/epl_features_2021_2024.csv')
    features_df['Date'] = pd.to_datetime(features_df['Date'])
    features_df = features_df.sort_values('Date')
    
    return features_df


def time_based_split(df, train_frac=0.70, val_frac=0.15):
    """Split data by time."""
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    
    return train, val, test


def get_poisson_predictions(features_df, split='test'):
    """Get Poisson model predictions."""
    logger.info("Getting Poisson predictions...")
    
    # Load trained model
    model = PoissonMatchPredictor.load('models/poisson_v1_latest.pkl')
    
    # Split data
    train_df, val_df, test_df = time_based_split(features_df)
    
    if split == 'val':
        df = val_df
    elif split == 'test':
        df = test_df
    else:
        df = train_df
    
    # Rename columns for Poisson
    df_poisson = df.copy()
    if 'home_team' not in df_poisson.columns:
        df_poisson = df_poisson.rename(columns={'HomeTeam': 'home_team', 'AwayTeam': 'away_team'})
    
    #  Capitalize for Poisson
    df_poisson = df_poisson.rename(columns={'home_team': 'HomeTeam', 'away_team': 'AwayTeam'})
    
    # Predict
    predictions = model.predict(df_poisson)
    
    return predictions[['prob_home', 'prob_draw', 'prob_away']].values


def get_catboost_predictions(features_df, split='test'):
    """Get CatBoost model predictions."""
    logger.info("Getting CatBoost predictions...")
    
    # Load model
    loader = ModelLoader()
    loader.load_latest()
    
    # Split
    train_df, val_df, test_df = time_based_split(features_df)
    
    if split == 'val':
        df = val_df
    elif split == 'test':
        df = test_df
    else:
        df = train_df
    
    # Get features
    feature_cols = [col for col in df.columns 
                   if col not in ['Date', 'HomeTeam', 'AwayTeam', 'Season', 'FTR']]
    X = df[feature_cols].values
    
    # Predict
    predictions = loader.predict(X)
    
    return predictions


def get_neural_predictions(features_df, split='test'):
    """Get Neural model predictions."""
    logger.info("Getting Neural predictions...")
    
    import torch
    
    # Load model
    checkpoint = torch.load('models/neural_v1_latest.pt', weights_only=False)
    
    model = DenseNN(input_dim=22, hidden_dims=[64, 32, 16], dropout=0.3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler = checkpoint['scaler']
    calibrators = checkpoint['calibrators']
    
    # Split
    train_df, val_df, test_df = time_based_split(features_df)
    
    if split == 'val':
        df = val_df
    elif split == 'test':
        df = test_df
    else:
        df = train_df
    
    # Get features
    feature_cols = [col for col in df.columns 
                   if col not in ['Date', 'HomeTeam', 'AwayTeam', 'Season', 'FTR']]
    X = df[feature_cols].values
    
    # Scale and predict
    X_scaled = scaler.transform(X)
    X_tensor = torch.FloatTensor(X_scaled)
    
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).numpy()
    
    # Apply calibration
    if calibrators:
        calibrated = np.zeros_like(probs)
        for i, cal in enumerate(calibrators):
            calibrated[:, i] = cal.predict(probs[:, i])
        
        row_sums = calibrated.sum(axis=1, keepdims=True)
        probs = calibrated / row_sums
    
    return probs


def compile_meta_dataset(split='test'):
    """
    Compile meta-dataset for meta-model training.
    
    Returns:
        X_meta: Meta-features (n_samples, 9)
        y: True labels (n_samples,)
    """
    logger.info("=" * 70)
    logger.info(f"COMPILING META-DATASET ({split} set)")
    logger.info("=" * 70)
    
    # Load data
    features_df = load_data()
    
    # Get predictions from all models
    poisson_preds = get_poisson_predictions(features_df, split)
    catboost_preds = get_catboost_predictions(features_df, split)
    neural_preds = get_neural_predictions(features_df, split)
    
    logger.info(f"  Poisson predictions: {poisson_preds.shape}")
    logger.info(f"  CatBoost predictions: {catboost_preds.shape}")
    logger.info(f"  Neural predictions: {neural_preds.shape}")
    
    # Align lengths (take minimum)
    min_len = min(len(poisson_preds), len(catboost_preds), len(neural_preds))
    
    poisson_preds = poisson_preds[:min_len]
    catboost_preds = catboost_preds[:min_len]
    neural_preds = neural_preds[:min_len]
    
    logger.info(f"  Aligned to {min_len} samples")
    
    # Create meta-features
    X_meta = np.hstack([
        poisson_preds,   # 3 features
        catboost_preds,  # 3 features
        neural_preds     # 3 features
    ])
    
    # Get true labels
    _, val_df, test_df = time_based_split(features_df)
    
    if split == 'val':
        df = val_df
    elif split == 'test':
        df = test_df
    else:
        _, val_df, test_df = time_based_split(features_df)
        df = features_df.iloc[:int(len(features_df) * 0.70)]
    
    result_map = {'H': 0, 'D': 1, 'A': 2}
    y = df['FTR'].map(result_map).values[:min_len]
    
    logger.info(f"\n✓ Meta-dataset compiled:")
    logger.info(f"  Features: {X_meta.shape}")
    logger.info(f"  Labels: {y.shape}")
    logger.info(f"  Feature names: [poisson_H, poisson_D, poisson_A, "
                f"catboost_H, catboost_D, catboost_A, "
                f"neural_H, neural_D, neural_A]")
    
    return X_meta, y


def main():
    """Compile and save meta-datasets."""
    output_dir = Path('data/meta/')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compile for all splits
    for split in ['train', 'val', 'test']:
        X_meta, y = compile_meta_dataset(split)
        
        # Save
        np.save(output_dir / f'X_meta_{split}.npy', X_meta)
        np.save(output_dir / f'y_meta_{split}.npy', y)
        
        logger.info(f"✓ Saved {split} meta-dataset\n")
    
    logger.info("=" * 70)
    logger.info("✅ ALL META-DATASETS COMPILED")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
