"""
ML model for football match outcome prediction using LightGBM.

Implements gradient boosting with temporal train/valid split and
probability calibration to ensure well-calibrated predictions.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from ..logging_setup import get_logger

logger = get_logger(__name__)


# Feature columns used for training
FEATURE_COLS = [
    'home_goals_for_avg_5',
    'home_goals_against_avg_5',
    'home_points_avg_5',
    'home_form_points_5',
    'home_matches_count',
    'away_goals_for_avg_5',
    'away_goals_against_avg_5',
    'away_points_avg_5',
    'away_form_points_5',
    'away_matches_count',
]


def create_target(df: pd.DataFrame) -> np.ndarray:
    """
    Create target variable from actual match results.
    
    Args:
        df: DataFrame with home_goals and away_goals
        
    Returns:
        Target array: 0=Away win, 1=Draw, 2=Home win
    """
    home_goals = df['home_goals'].values
    away_goals = df['away_goals'].values
    
    # Target encoding:
    # - Home win (home_goals > away_goals) → 2
    # - Draw (home_goals == away_goals) → 1
    # - Away win (home_goals < away_goals) → 0
    target = np.where(
        home_goals > away_goals,
        2,  # Home win
        np.where(
            home_goals == away_goals,
            1,  # Draw
            0   # Away win
        )
    )
    
    return target


def temporal_train_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by date for temporal validation.
    
    CRITICAL: NO SHUFFLE! Maintains temporal ordering to prevent data leakage.
    
    Args:
        df: DataFrame with 'date' column
        train_ratio: Fraction of data for training (default: 0.7)
        
    Returns:
        (train_df, valid_df) split by date
    """
    # Sort by date to ensure temporal ordering
    df_sorted = df.sort_values('date').reset_index(drop=True)
    
    # Calculate split point
    split_idx = int(len(df_sorted) * train_ratio)
    
    train_df = df_sorted.iloc[:split_idx].copy()
    valid_df = df_sorted.iloc[split_idx:].copy()
    
    # Verify temporal split
    max_train_date = train_df['date'].max()
    min_valid_date = valid_df['date'].min()
    
    logger.info(
        f"Temporal split: train up to {max_train_date}, "
        f"valid from {min_valid_date}"
    )
    
    if max_train_date > min_valid_date:
        raise ValueError(
            "Temporal split violation: train dates overlap with valid dates!"
        )
    
    return train_df, valid_df


class MLModel:
    """
    ML model for match outcome prediction using LightGBM.
    
    Features:
    - Multiclass classification (Home/Draw/Away)
    - Temporal train/valid split
    - Probability calibration
    - Model persistence
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        random_state: int = 42,
        calibration_method: str = 'isotonic'
    ):
        """
        Initialize ML model.
        
        Args:
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate
            max_depth: Maximum tree depth
            random_state: Random seed
            calibration_method: 'isotonic' or 'sigmoid' (Platt scaling)
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.calibration_method = calibration_method
        
        self.base_model: Optional[lgb.LGBMClassifier] = None
        self.calibrated_model: Optional[CalibratedClassifierCV] = None
        self.feature_cols = FEATURE_COLS
        
        logger.info(
            f"Initialized MLModel: n_estimators={n_estimators}, "
            f"lr={learning_rate}, max_depth={max_depth}, "
            f"calibration={calibration_method}"
        )
    
    def train(
        self,
        features_df: pd.DataFrame,
        train_ratio: float = 0.7
    ) -> dict:
        logger.info(f"Training ML model on {len(features_df)} matches")

        df = features_df.copy()

                # ---------- target creation ----------
        if "target" not in df.columns:
            if {"home_goals", "away_goals"}.issubset(df.columns):
                df["target"] = np.where(
                    df["home_goals"] > df["away_goals"], 2,
                    np.where(df["home_goals"] < df["away_goals"], 0, 1)
                )
            else:
                raise ValueError(
                    "features_df must contain either 'target' "
                    "or ('home_goals' and 'away_goals') columns"
                )

        from sklearn.model_selection import train_test_split

        # ---------- split ----------
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

            if df["date"].isna().all():
                train_df, valid_df = train_test_split(
                    df, train_size=train_ratio, shuffle=True, random_state=42
                )
            else:
                train_df, valid_df = temporal_train_test_split(df, train_ratio)
        else:
            train_df, valid_df = train_test_split(
                df, train_size=train_ratio, shuffle=True, random_state=42
            )

        logger.info(f"Train: {len(train_df)}, Valid: {len(valid_df)}")

        # ---------- X / y ----------
        X_train = train_df[self.feature_cols].fillna(0.0)
        y_train = train_df["target"].values

        X_valid = valid_df[self.feature_cols].fillna(0.0)
        y_valid = valid_df["target"].values

        # ---------- model ----------
        base_model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            objective="multiclass",
            num_class=3,
            random_state=42,
        )

        # CRITICAL: Save base_model reference before calibration
        self.base_model = base_model

        self.calibrated_model = CalibratedClassifierCV(
            base_model,
            method=self.calibration_method,
            cv=3,
        )

        self.calibrated_model.fit(X_train, y_train)

        # ---------- validation ----------
        train_pred = self.calibrated_model.predict(X_train)
        train_acc = (train_pred == y_train).mean()
        
        valid_pred = self.calibrated_model.predict(X_valid)
        valid_acc = (valid_pred == y_valid).mean()

        logger.info(f"Training complete: valid_acc={valid_acc:.3f}")

        return {
            "train_samples": len(train_df),
            "valid_samples": len(valid_df),
            "train_accuracy": float(train_acc),
            "valid_accuracy": float(valid_acc),
            "train_accuracy_cal": float(train_acc),
            "valid_accuracy_cal": float(valid_acc),
        }



    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate calibrated predictions for matches.
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            DataFrame with original features plus ML predictions
        """
        if self.calibrated_model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info(f"Generating ML predictions for {len(features_df)} matches")
        
        # Prepare features (keep as DataFrame to preserve feature names)
        X = features_df[self.feature_cols].fillna(0.0)
        
        # Get probabilities (calibrated if available, else raw model)
        model = self.calibrated_model if self.calibrated_model is not None else self.base_model
        probs = model.predict_proba(X)
        
        # Add predictions to dataframe
        predictions_df = features_df.copy()
        predictions_df['prob_away_ml'] = probs[:, 0]  # Away win
        predictions_df['prob_draw_ml'] = probs[:, 1]  # Draw
        predictions_df['prob_home_ml'] = probs[:, 2]  # Home win
        
        # Verify probabilities sum to 1.0
        prob_sums = probs.sum(axis=1)
        max_deviation = abs(prob_sums - 1.0).max()
        
        logger.info(f"Maximum probability sum deviation: {max_deviation:.6f}")
        
        if max_deviation > 0.001:
            logger.warning("Some probabilities deviate from 1.0 by more than 0.001")
        
        return predictions_df
    
    def save(self, filepath: Path) -> None:
        """
        Save trained model to file.
        
        Args:
            filepath: Path to save model (pickle format)
        """
        if self.calibrated_model is None:
            raise ValueError("No model to save. Train first.")
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            # calibrated_model can be None (we may skip calibration on tiny/degenerate valid sets)
            'calibrated_model': self.calibrated_model,
            'base_model': self.base_model,
            'feature_cols': self.feature_cols,
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'calibration_method': self.calibration_method,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path) -> 'MLModel':
        """
        Load trained model from file.
        
        Args:
            filepath: Path to saved model
            
        Returns:
            Loaded MLModel instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Reconstruct model
        model = cls(
            n_estimators=model_data['n_estimators'],
            learning_rate=model_data['learning_rate'],
            max_depth=model_data['max_depth'],
            calibration_method=model_data['calibration_method']
        )
        
        model.calibrated_model = model_data['calibrated_model']
        model.base_model = model_data['base_model']
        model.feature_cols = model_data['feature_cols']
        
        logger.info(f"Model loaded from {filepath}")
        
        return model
    
    def train_from_file(
        self,
        input_file: Path,
        output_file: Path,
        train_ratio: float = 0.7
    ) -> Dict[str, any]:
        """
        Train model from features file and save.
        
        Args:
            input_file: Path to features CSV
            output_file: Path to save model
            train_ratio: Train/valid split ratio
            
        Returns:
            Training statistics
        """
        logger.info(f"Loading features from {input_file}")
        features_df = pd.read_csv(input_file)

        # Convert date to datetime if the column exists
        if "date" in features_df.columns:
            features_df["date"] = pd.to_datetime(features_df["date"], errors="coerce")

        
        # Train model
        stats = self.train(features_df, train_ratio)
        
        # Save model
        self.save(output_file)
        
        return stats
    
    def predict_from_file(
        self,
        input_file: Path,
        output_file: Path
    ) -> Dict[str, any]:
        """
        Load features, predict, and save results.
        
        Args:
            input_file: Path to features CSV
            output_file: Path to save predictions CSV
            
        Returns:
            Prediction statistics
        """
        logger.info(f"Loading features from {input_file}")
        features_df = pd.read_csv(input_file)
        
        # Convert date to datetime if present
        if 'date' in features_df.columns:
            features_df['date'] = pd.to_datetime(features_df['date'], errors='coerce')
        
        # Generate predictions
        predictions_df = self.predict(features_df)
        
        # Save to file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        predictions_df.to_csv(output_file, index=False)
        
        logger.info(f"Saved predictions to {output_file}")
        
        # Calculate statistics
        stats = {
            'total_matches': len(predictions_df),
            'avg_prob_home': predictions_df['prob_home_ml'].mean(),
            'avg_prob_draw': predictions_df['prob_draw_ml'].mean(),
            'avg_prob_away': predictions_df['prob_away_ml'].mean(),
        }
        
        return stats
