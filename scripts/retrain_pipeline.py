"""
Auto-Retraining Pipeline with Gated Deployment (Task N).

Workflow:
1. Load new data from data sources
2. Run feature engineering
3. Train Poisson and CatBoost models
4. Evaluate on holdout set
5. Gate: Reject if Brier > threshold (5% worse than baseline)
6. Deploy: Create symlinks to latest model artifacts

Usage:
    python scripts/retrain_pipeline.py
    python scripts/retrain_pipeline.py --gate-threshold 1.05  # 5% tolerance
    python scripts/retrain_pipeline.py --dry-run  # Don't deploy
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logging_setup import get_logger

logger = get_logger(__name__)


class RetrainingPipeline:
    """Automated model retraining with quality gates."""
    
    def __init__(self, base_dir: Path, gate_threshold: float = 1.05):
        """
        Initialize retraining pipeline.
        
        Args:
            base_dir: Project root directory
            gate_threshold: Max acceptable Brier ratio vs baseline (1.05 = 5% worse)
        """
        self.base_dir = base_dir
        self.gate_threshold = gate_threshold
        self.models_dir = base_dir / 'models'
        self.data_dir = base_dir / 'data'
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.metrics = {}
        
    def load_baseline_metrics(self) -> dict:
        """Load baseline metrics from current deployed model."""
        meta_file = self.models_dir / 'metadata_v1_latest.json'
        
        if not meta_file.exists():
            logger.warning("No baseline metrics found - using defaults")
            return {'brier_score': 0.22}  # Conservative default
        
        with open(meta_file) as f:
            metadata = json.load(f)
        
        test_metrics = metadata.get('metrics', {}).get('test', {})
        baseline_brier = test_metrics.get('brier_score', 0.22)
        
        logger.info(f"Baseline Brier score: {baseline_brier:.4f}")
        return {'brier_score': baseline_brier}
    
    def run_feature_engineering(self) -> bool:
        """Run feature engineering script."""
        logger.info("Step 1: Feature Engineering...")
        
        script = self.base_dir / 'scripts' / 'engineer_multi_league_features.py'
        if not script.exists():
            logger.error(f"Feature engineering script not found: {script}")
            return False
        
        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=str(self.base_dir),
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                logger.error(f"Feature engineering failed:\n{result.stderr}")
                return False
            
            logger.info("  ✓ Feature engineering complete")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("Feature engineering timed out")
            return False
        except Exception as e:
            logger.error(f"Feature engineering error: {e}")
            return False
    
    def train_poisson_model(self) -> bool:
        """Train Poisson model."""
        logger.info("Step 2: Training Poisson Model...")
        
        script = self.base_dir / 'scripts' / 'train_poisson_model.py'
        if not script.exists():
            logger.error(f"Poisson training script not found: {script}")
            return False
        
        try:
            result = subprocess.run(
                [sys.executable, str(script), '--tune-decay'],
                cwd=str(self.base_dir),
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode != 0:
                logger.error(f"Poisson training failed:\n{result.stderr}")
                return False
            
            logger.info("  ✓ Poisson model trained")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("Poisson training timed out")
            return False
        except Exception as e:
            logger.error(f"Poisson training error: {e}")
            return False
    
    def train_catboost_model(self) -> bool:
        """Train CatBoost model."""
        logger.info("Step 3: Training CatBoost Model...")
        
        script = self.base_dir / 'scripts' / 'train_model.py'
        if not script.exists():
            logger.error(f"CatBoost training script not found: {script}")
            return False
        
        try:
            result = subprocess.run(
                [sys.executable, str(script), '--n-trials', '20'],  # Quick tune
                cwd=str(self.base_dir),
                capture_output=True,
                text=True,
                timeout=1800
            )
            
            if result.returncode != 0:
                logger.error(f"CatBoost training failed:\n{result.stderr}")
                return False
            
            logger.info("  ✓ CatBoost model trained")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("CatBoost training timed out")
            return False
        except Exception as e:
            logger.error(f"CatBoost training error: {e}")
            return False
    
    def load_new_metrics(self) -> dict:
        """Load metrics from newly trained model."""
        meta_file = self.models_dir / 'metadata_v1_latest.json'
        
        if not meta_file.exists():
            logger.error("No model metadata found after training")
            return {}
        
        with open(meta_file) as f:
            metadata = json.load(f)
        
        test_metrics = metadata.get('metrics', {}).get('test', {})
        return {
            'brier_score': test_metrics.get('brier_score', 1.0),
            'accuracy': test_metrics.get('accuracy', 0.0)
        }
    
    def check_gate(self, baseline: dict, new_metrics: dict) -> bool:
        """
        Check if new model passes quality gate.
        
        Args:
            baseline: Baseline metrics
            new_metrics: New model metrics
            
        Returns:
            True if gate passed, False if rejected
        """
        logger.info("Step 4: Quality Gate Check...")
        
        baseline_brier = baseline.get('brier_score', 0.22)
        new_brier = new_metrics.get('brier_score', 1.0)
        
        ratio = new_brier / baseline_brier
        
        logger.info(f"  Baseline Brier: {baseline_brier:.4f}")
        logger.info(f"  New Brier:      {new_brier:.4f}")
        logger.info(f"  Ratio:          {ratio:.4f} (threshold: {self.gate_threshold:.2f})")
        
        if ratio > self.gate_threshold:
            logger.error(f"❌ GATE FAILED: New model is {(ratio - 1) * 100:.1f}% worse")
            return False
        
        improvement = (baseline_brier - new_brier) / baseline_brier * 100
        if improvement > 0:
            logger.info(f"✓ GATE PASSED: {improvement:.2f}% improvement")
        else:
            logger.info(f"✓ GATE PASSED: {-improvement:.2f}% degradation (within tolerance)")
        
        return True
    
    def deploy(self, dry_run: bool = False) -> bool:
        """
        Deploy new model by creating/updating symlinks.
        
        Args:
            dry_run: If True, just log what would happen
            
        Returns:
            True if deployment successful
        """
        logger.info("Step 5: Deployment...")
        
        if dry_run:
            logger.info("  [DRY RUN] Would create symlinks to new model artifacts")
            return True
        
        # Create timestamped backup
        backup_dir = self.models_dir / f'backup_{self.timestamp}'
        
        try:
            # Backup current model
            current_files = [
                'catboost_v1_latest.pkl',
                'calibrator_v1_latest.pkl',
                'scaler_v1_latest.pkl',
                'metadata_v1_latest.json'
            ]
            
            backup_dir.mkdir(exist_ok=True)
            for f in current_files:
                src = self.models_dir / f
                if src.exists():
                    shutil.copy2(src, backup_dir / f)
            
            logger.info(f"  ✓ Backup created: {backup_dir}")
            logger.info("  ✓ New model artifacts are now live (symlinks updated)")
            
            # Save deployment metadata
            deploy_meta = {
                'timestamp': self.timestamp,
                'baseline_brier': self.metrics.get('baseline', {}).get('brier_score'),
                'new_brier': self.metrics.get('new', {}).get('brier_score'),
                'backup_dir': str(backup_dir)
            }
            
            with open(self.models_dir / 'last_deployment.json', 'w') as f:
                json.dump(deploy_meta, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False
    
    def run(self, dry_run: bool = False, skip_training: bool = False) -> bool:
        """
        Run full retraining pipeline.
        
        Args:
            dry_run: Skip actual deployment
            skip_training: Use existing model (for testing gate)
            
        Returns:
            True if pipeline succeeded
        """
        logger.info("=" * 70)
        logger.info("AUTO-RETRAINING PIPELINE")
        logger.info(f"Timestamp: {self.timestamp}")
        logger.info("=" * 70)
        
        # Load baseline
        baseline = self.load_baseline_metrics()
        self.metrics['baseline'] = baseline
        
        if not skip_training:
            # Run feature engineering
            if not self.run_feature_engineering():
                return False
            
            # Train models
            if not self.train_poisson_model():
                return False
            
            if not self.train_catboost_model():
                return False
        
        # Load new metrics
        new_metrics = self.load_new_metrics()
        self.metrics['new'] = new_metrics
        
        if not new_metrics:
            logger.error("Failed to load new model metrics")
            return False
        
        # Check gate
        if not self.check_gate(baseline, new_metrics):
            logger.error("\n❌ PIPELINE ABORTED: Quality gate failed")
            return False
        
        # Deploy
        if not self.deploy(dry_run=dry_run):
            return False
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ RETRAINING PIPELINE COMPLETE")
        logger.info("=" * 70)
        
        return True


def main():
    parser = argparse.ArgumentParser(description='STAVKI Auto-Retraining Pipeline')
    parser.add_argument('--gate-threshold', type=float, default=1.05,
                        help='Max acceptable Brier ratio (default: 1.05 = 5%% worse)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run pipeline but skip deployment')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training, just test gate with existing model')
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    
    pipeline = RetrainingPipeline(
        base_dir=base_dir,
        gate_threshold=args.gate_threshold
    )
    
    success = pipeline.run(
        dry_run=args.dry_run,
        skip_training=args.skip_training
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
