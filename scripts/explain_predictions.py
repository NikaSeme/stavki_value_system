"""
SHAP Explainability for STAVKI Predictions (Task R).

Provides:
- SHAP TreeExplainer for CatBoost model
- Summary plots showing feature importance
- Per-prediction explanations
- Feature importance reports

Usage:
    python scripts/explain_predictions.py
    python scripts/explain_predictions.py --top 20  # Show top 20 features
    python scripts/explain_predictions.py --match "Man City vs Liverpool"  # Explain specific match
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging_setup import get_logger

logger = get_logger(__name__)

# Optional imports - graceful degradation if SHAP not installed
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed. Run: pip install shap")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class PredictionExplainer:
    """
    Explain STAVKI predictions using SHAP values.
    """
    
    def __init__(self, model, feature_names: list = None):
        """
        Initialize explainer with trained model.
        
        Args:
            model: Trained CatBoost or compatible model
            feature_names: List of feature names
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required but not installed")
        
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self._init_explainer()
        
    def _init_explainer(self):
        """Initialize SHAP TreeExplainer."""
        try:
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("✓ SHAP TreeExplainer initialized")
        except Exception as e:
            logger.warning(f"TreeExplainer failed, trying Explainer: {e}")
            self.explainer = shap.Explainer(self.model)
    
    def compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for input features.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            SHAP values array
        """
        shap_values = self.explainer.shap_values(X)
        return shap_values
    
    def get_feature_importance(self, X: np.ndarray, top_n: int = 20) -> pd.DataFrame:
        """
        Get global feature importance from SHAP values.
        
        Args:
            X: Feature matrix
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        shap_values = self.compute_shap_values(X)
        
        # For multi-class, average across classes
        if isinstance(shap_values, list):
            # Average importance across classes
            importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            importance = np.abs(shap_values).mean(axis=0)
        
        # Create DataFrame
        df = pd.DataFrame({
            'feature': self.feature_names if self.feature_names else [f'f_{i}' for i in range(len(importance))],
            'importance': importance
        })
        
        df = df.sort_values('importance', ascending=False).head(top_n)
        df['rank'] = range(1, len(df) + 1)
        
        return df[['rank', 'feature', 'importance']]
    
    def explain_prediction(
        self,
        X_single: np.ndarray,
        match_info: dict = None,
        top_n: int = 10
    ) -> dict:
        """
        Explain a single prediction.
        
        Args:
            X_single: Single sample feature vector
            match_info: Optional match metadata
            top_n: Number of top features to show
            
        Returns:
            Dict with explanation details
        """
        if X_single.ndim == 1:
            X_single = X_single.reshape(1, -1)
        
        shap_values = self.compute_shap_values(X_single)
        
        # Get model prediction
        proba = self.model.predict_proba(X_single)[0]
        pred_class = np.argmax(proba)
        
        class_names = ['Home', 'Draw', 'Away']
        
        # Get SHAP values for predicted class
        if isinstance(shap_values, list):
            sv = shap_values[pred_class][0]
        else:
            sv = shap_values[0]
        
        # Get top contributing features
        feature_names = self.feature_names if self.feature_names else [f'f_{i}' for i in range(len(sv))]
        
        contributions = sorted(
            zip(feature_names, sv, X_single[0]),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_n]
        
        explanation = {
            'prediction': class_names[pred_class],
            'probabilities': {
                'home': float(proba[0]),
                'draw': float(proba[1]),
                'away': float(proba[2])
            },
            'confidence': float(max(proba)),
            'top_factors': [
                {
                    'feature': name,
                    'shap_value': float(shap_value),
                    'feature_value': float(value),
                    'direction': 'increases' if shap_value > 0 else 'decreases'
                }
                for name, shap_value, value in contributions
            ]
        }
        
        if match_info:
            explanation['match'] = match_info
        
        return explanation
    
    def generate_summary_plot(
        self,
        X: np.ndarray,
        output_path: Path = None,
        class_index: int = 0
    ):
        """
        Generate SHAP summary plot.
        
        Args:
            X: Feature matrix
            output_path: Path to save plot (optional)
            class_index: Class index for multi-class (0=Home, 1=Draw, 2=Away)
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available - skipping plot")
            return
        
        shap_values = self.compute_shap_values(X)
        
        # Handle multi-class
        if isinstance(shap_values, list):
            sv = shap_values[class_index]
        else:
            sv = shap_values
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            sv,
            X,
            feature_names=self.feature_names,
            show=False
        )
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Summary plot saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_importance_report(
        self,
        X: np.ndarray,
        output_path: Path,
        top_n: int = 50
    ):
        """
        Save feature importance report to JSON.
        
        Args:
            X: Feature matrix
            output_path: Path to save report
            top_n: Number of features to include
        """
        importance_df = self.get_feature_importance(X, top_n=top_n)
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'n_samples': len(X),
            'n_features': X.shape[1] if len(X.shape) > 1 else 1,
            'top_features': importance_df.to_dict('records')
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Feature importance report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='STAVKI Prediction Explainer')
    parser.add_argument('--top', type=int, default=20,
                        help='Number of top features to show')
    parser.add_argument('--output-dir', type=str, default='reports',
                        help='Directory for output files')
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("SHAP PREDICTION EXPLAINER")
    logger.info("=" * 70)
    
    if not SHAP_AVAILABLE:
        logger.error("SHAP is not installed. Run: pip install shap")
        return
    
    # Load model
    base_dir = Path(__file__).parent.parent
    model_path = base_dir / 'models' / 'catboost_v1_latest.pkl'
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.info("Please train a model first: python scripts/train_model.py")
        return
    
    import pickle
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    feature_names = model_data.get('feature_names', None)
    
    logger.info(f"✓ Model loaded: {model_path}")
    
    # Load sample data
    data_path = base_dir / 'data' / 'processed' / 'multi_league_features_2021_2024.csv'
    
    if not data_path.exists():
        logger.error(f"Data not found: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    
    # Get features
    exclude_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG', 
                    'Season', 'League', 'GoalDiff', 'TotalGoals']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].fillna(0).values[-1000:]  # Last 1000 samples
    
    logger.info(f"Analyzing {len(X)} samples with {len(feature_cols)} features")
    
    # Create explainer
    explainer = PredictionExplainer(model, feature_names=feature_cols)
    
    # Get feature importance
    logger.info(f"\nTop {args.top} Features:")
    importance_df = explainer.get_feature_importance(X, top_n=args.top)
    
    for _, row in importance_df.iterrows():
        logger.info(f"  {row['rank']:2d}. {row['feature']:<30} {row['importance']:.4f}")
    
    # Save outputs
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)
    
    # Save importance report
    explainer.save_importance_report(
        X,
        output_dir / 'feature_importance.json',
        top_n=args.top
    )
    
    # Generate summary plot
    if MATPLOTLIB_AVAILABLE:
        explainer.generate_summary_plot(
            X,
            output_path=output_dir / 'shap_summary.png',
            class_index=0
        )
    
    # Example single prediction explanation
    sample_idx = -1  # Latest match
    explanation = explainer.explain_prediction(X[sample_idx])
    
    logger.info(f"\nExample Prediction Explanation:")
    logger.info(f"  Prediction: {explanation['prediction']}")
    logger.info(f"  Confidence: {explanation['confidence']:.1%}")
    logger.info(f"  Top Factors:")
    for factor in explanation['top_factors'][:5]:
        logger.info(f"    - {factor['feature']}: {factor['shap_value']:+.3f} ({factor['direction']} prob)")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ SHAP ANALYSIS COMPLETE")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
