"""
Time-Series Cross-Validation for STAVKI

Provides time-aware cross-validation splitters that ensure:
- Training data always precedes test data chronologically
- No future information leakage
- Progressive training set expansion across folds
"""

from typing import Iterator, List, Tuple, Optional
import numpy as np
import pandas as pd
from src.logging_setup import get_logger

logger = get_logger(__name__)


class TimeSeriesFold:
    """
    Time-series aware cross-validation splitter.
    
    Unlike standard K-fold, this ensures temporal ordering where
    training data always occurs before test data. This is critical
    for sports betting predictions where using future data would
    cause severe data leakage.
    
    The splitter uses an expanding window approach:
    - Fold 1: Train on early data, test on next chunk
    - Fold 2: Train on fold 1 train + test, test on next chunk
    - etc.
    
    Attributes:
        n_splits: Number of folds for cross-validation
        test_size: Fraction of data to use for test in each fold
        gap: Number of samples to skip between train and test (optional)
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: float = 0.15,
        gap: int = 0
    ):
        """
        Initialize TimeSeriesFold.
        
        Args:
            n_splits: Number of CV folds (default: 5)
            test_size: Fraction of data for test in each fold (default: 0.15)
            gap: Number of samples to skip between train and test to prevent
                 leakage from features computed on recent matches (default: 0)
        """
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if gap < 0:
            raise ValueError("gap must be non-negative")
            
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
    
    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[List[int], List[int]]]:
        """
        Generate train/test indices for time-series cross-validation.
        
        The data is split into n_splits + 1 chunks. For each fold i:
        - Train uses chunks 0 to i (expanding window)
        - Test uses chunk i + 1
        
        Args:
            X: Features array or DataFrame (used for length only)
            y: Labels (unused, for sklearn compatibility)
            groups: Group labels (unused, for sklearn compatibility)
            
        Yields:
            Tuple of (train_indices, test_indices) for each fold
        """
        n_samples = len(X) if hasattr(X, '__len__') else X.shape[0]
        test_size = int(n_samples * self.test_size)
        
        if test_size == 0:
            raise ValueError(f"test_size {self.test_size} results in 0 test samples for {n_samples} total samples")
        
        logger.info(f"TimeSeriesFold: {n_samples} samples, {self.n_splits} folds, ~{test_size} test samples/fold")
        
        for i in range(self.n_splits):
            # Calculate train and test boundaries for this fold
            # Expanding window: train gets progressively larger
            train_end = n_samples - (self.n_splits - i) * test_size - self.gap
            test_start = train_end + self.gap
            test_end = min(test_start + test_size, n_samples)
            
            if train_end <= 0:
                logger.warning(f"Fold {i+1}: Insufficient training data, skipping")
                continue
                
            if test_start >= n_samples:
                logger.warning(f"Fold {i+1}: No test data available, skipping")
                continue
            
            train_indices = list(range(0, train_end))
            test_indices = list(range(test_start, test_end))
            
            logger.debug(f"Fold {i+1}: Train [{0}:{train_end}] ({len(train_indices)} samples), "
                        f"Test [{test_start}:{test_end}] ({len(test_indices)} samples)")
            
            yield train_indices, test_indices
    
    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> int:
        """Return number of splits (for sklearn compatibility)."""
        return self.n_splits


class SlidingWindowFold:
    """
    Sliding window cross-validation for time-series.
    
    Unlike TimeSeriesFold (expanding window), this uses a fixed-size
    training window that slides forward through time.
    
    Useful when:
    - You want equal training set sizes across folds
    - Recent data is more relevant than older data
    - You want faster iterations with smaller training sets
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        train_size: float = 0.50,
        test_size: float = 0.10,
        gap: int = 0
    ):
        """
        Initialize SlidingWindowFold.
        
        Args:
            n_splits: Number of CV folds
            train_size: Fraction of data for training window (fixed size)
            test_size: Fraction of data for test in each fold
            gap: Samples to skip between train and test
        """
        if train_size + test_size > 1:
            raise ValueError("train_size + test_size must not exceed 1")
            
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.gap = gap
    
    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[List[int], List[int]]]:
        """Generate train/test indices with sliding window."""
        n_samples = len(X) if hasattr(X, '__len__') else X.shape[0]
        train_size = int(n_samples * self.train_size)
        test_size = int(n_samples * self.test_size)
        
        # Calculate step size to create n_splits folds
        available = n_samples - train_size - test_size - self.gap
        step = max(1, available // max(1, self.n_splits - 1)) if self.n_splits > 1 else 0
        
        logger.info(f"SlidingWindowFold: {n_samples} samples, window={train_size}, step={step}")
        
        for i in range(self.n_splits):
            train_start = i * step
            train_end = train_start + train_size
            test_start = train_end + self.gap
            test_end = min(test_start + test_size, n_samples)
            
            if test_end > n_samples:
                break
                
            train_indices = list(range(train_start, train_end))
            test_indices = list(range(test_start, test_end))
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


def cross_val_score_timeseries(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: TimeSeriesFold,
    scoring: str = 'brier'
) -> List[float]:
    """
    Calculate cross-validation scores for time-series data.
    
    Args:
        model: Scikit-learn compatible model with fit/predict_proba
        X: Features
        y: Labels (0, 1, 2 for 3-way classification)
        cv: TimeSeriesFold or SlidingWindowFold instance
        scoring: Metric to compute ('brier', 'accuracy', 'logloss')
        
    Returns:
        List of scores for each fold
    """
    from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
    
    scores = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Clone and fit model
        from sklearn.base import clone
        fold_model = clone(model)
        fold_model.fit(X_train, y_train)
        
        # Predict
        probs = fold_model.predict_proba(X_test)
        preds = probs.argmax(axis=1)
        
        # Calculate score
        if scoring == 'brier':
            # Average Brier across classes
            briers = []
            for i in range(3):
                y_binary = (y_test == i).astype(int)
                if len(np.unique(y_binary)) > 1:  # Both classes present
                    briers.append(brier_score_loss(y_binary, probs[:, i]))
            score = np.mean(briers) if briers else 0.0
        elif scoring == 'accuracy':
            score = accuracy_score(y_test, preds)
        elif scoring == 'logloss':
            score = log_loss(y_test, probs)
        else:
            raise ValueError(f"Unknown scoring: {scoring}")
        
        scores.append(score)
        logger.info(f"Fold {fold_idx + 1}: {scoring}={score:.4f}")
    
    return scores


if __name__ == '__main__':
    # Example usage
    import numpy as np
    
    # Create dummy time-series data
    n_samples = 1000
    X = np.random.randn(n_samples, 10)
    y = np.random.randint(0, 3, n_samples)
    
    # Test TimeSeriesFold
    print("=" * 50)
    print("TimeSeriesFold Example")
    print("=" * 50)
    
    cv = TimeSeriesFold(n_splits=5, test_size=0.15)
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X), 1):
        print(f"Fold {fold}: Train={len(train_idx)} samples, Test={len(test_idx)} samples")
        print(f"  Train range: [{min(train_idx)}:{max(train_idx)+1}]")
        print(f"  Test range:  [{min(test_idx)}:{max(test_idx)+1}]")
        
        # Verify no overlap
        assert len(set(train_idx) & set(test_idx)) == 0, "Train/test overlap!"
        # Verify temporal ordering
        assert max(train_idx) < min(test_idx), "Train must precede test!"
    
    print("\n✓ All folds pass temporal ordering check")
