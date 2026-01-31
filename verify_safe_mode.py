
import sys
from pathlib import Path
import pandas as pd
import logging

# Add src to path
sys.path.append(str(Path.cwd()))

from src.models.ensemble_predictor import EnsemblePredictor

def verify_ensemble_safe_mode():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Initializing EnsemblePredictor (should utilize safe defaults)...")
    ensemble = EnsemblePredictor()
    
    if not ensemble.use_neural:
        logger.info("✅ Success: Neural model disabled by default.")
    else:
        logger.error("❌ Failure: Neural model is still enabled!")

    logger.info(f"Weights: {ensemble.weights}")
    
    if ensemble.weights['catboost'] == 0.70 and ensemble.weights['poisson'] == 0.30:
        logger.info("✅ Success: Weights adjusted correctly.")
    else:
        logger.error(f"❌ Failure: Unexpected weights: {ensemble.weights}")

if __name__ == "__main__":
    verify_ensemble_safe_mode()
