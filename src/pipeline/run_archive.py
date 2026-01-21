"""
Run archive system for STAVKI betting system.

Manages storage of run artifacts, metadata, and logs.
"""

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ..logging_setup import get_logger

logger = get_logger(__name__)


def get_git_commit() -> Optional[str]:
    """
    Get current git commit hash.
    
    Returns:
        Commit hash or None if not in git repo
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def create_run_directory(base_dir: Path = Path("runs")) -> Path:
    """
    Create timestamped run directory.
    
    Format: runs/YYYY-MM-DD/HHMMSS/
    
    Args:
        base_dir: Base directory for runs
        
    Returns:
        Path to created run directory
    """
    now = datetime.now()
    date_dir = base_dir / now.strftime("%Y-%m-%d")
    run_dir = date_dir / now.strftime("%H%M%S")
    
    run_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created run directory: {run_dir}")
    
    return run_dir


def save_run_metadata(
    run_dir: Path,
    bankroll: float,
    ev_threshold: float,
    kelly_fraction: float,
    max_stake_pct: float,
    max_bets: Optional[int] = None,
    model_version: str = "poisson_v1"
) -> None:
    """
    Save run metadata to JSON.
    
    Args:
        run_dir: Run directory
        bankroll: Bankroll amount
        ev_threshold: EV threshold
        kelly_fraction: Kelly fraction
        max_stake_pct: Max stake percentage
        max_bets: Max number of bets
        model_version: Model version identifier
    """
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'git_commit': get_git_commit(),
        'parameters': {
            'bankroll': bankroll,
            'ev_threshold': ev_threshold,
            'kelly_fraction': kelly_fraction,
            'max_stake_pct': max_stake_pct,
            'max_bets': max_bets,
            'model_version': model_version
        }
    }
    
    metadata_path = run_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved metadata: {metadata_path}")


def save_run_artifacts(
    run_dir: Path,
    predictions_df: Optional[pd.DataFrame] = None,
    recommendations_df: Optional[pd.DataFrame] = None,
    report: Optional[Dict] = None,
    report_text: Optional[str] = None
) -> Dict[str, Path]:
    """
    Save run artifacts (predictions, recommendations, reports).
    
    Args:
        run_dir: Run directory
        predictions_df: Predictions DataFrame
        recommendations_df: Recommendations DataFrame
        report: Report dictionary
        report_text: Report text
        
    Returns:
        Dict mapping artifact type to file path
    """
    paths = {}
    
    # Save predictions
    if predictions_df is not None and len(predictions_df) > 0:
        pred_csv = run_dir / "predictions.csv"
        pred_json = run_dir / "predictions.json"
        
        predictions_df.to_csv(pred_csv, index=False)
        predictions_df.to_json(pred_json, orient='records', indent=2)
        
        paths['predictions_csv'] = pred_csv
        paths['predictions_json'] = pred_json
        logger.info(f"Saved predictions: {pred_csv}")
    
    # Save recommendations
    if recommendations_df is not None and len(recommendations_df) > 0:
        rec_csv = run_dir / "recommendations.csv"
        rec_json = run_dir / "recommendations.json"
        
        recommendations_df.to_csv(rec_csv, index=False)
        recommendations_df.to_json(rec_json, orient='records', indent=2)
        
        paths['recommendations_csv'] = rec_csv
        paths['recommendations_json'] = rec_json
        logger.info(f"Saved recommendations: {rec_csv}")
    
    # Save report JSON
    if report is not None:
        report_json = run_dir / "report.json"
        with open(report_json, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        paths['report_json'] = report_json
        logger.info(f"Saved report JSON: {report_json}")
    
    # Save report text
    if report_text is not None:
        report_txt = run_dir / "report.txt"
        report_txt.write_text(report_text)
        paths['report_txt'] = report_txt
        logger.info(f"Saved report text: {report_txt}")
    
    return paths


def setup_run_logging(run_dir: Path) -> str:
    """
    Setup logging for this run.
    
    Creates run.log in the run directory.
    
    Args:
        run_dir: Run directory
        
    Returns:
        Path to run log file
    """
    import logging
    
    log_file = run_dir / "run.log"
    
    # Create file handler for this run
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    # Add to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    logger.info(f"Run logging configured: {log_file}")
    
    return str(log_file)


def list_runs(base_dir: Path = Path("runs")) -> list:
    """
    List all runs in chronological order.
    
    Args:
        base_dir: Base directory for runs
        
    Returns:
        List of run directories (newest first)
    """
    if not base_dir.exists():
        return []
    
    runs = []
    for date_dir in sorted(base_dir.iterdir(), reverse=True):
        if not date_dir.is_dir():
            continue
        for run_dir in sorted(date_dir.iterdir(), reverse=True):
            if run_dir.is_dir():
                runs.append(run_dir)
    
    return runs


def get_run_summary(run_dir: Path) -> Optional[Dict]:
    """
    Get summary of a run from its metadata and report.
    
    Args:
        run_dir: Run directory
        
    Returns:
        Summary dict or None if not found
    """
    metadata_path = run_dir / "metadata.json"
    report_path = run_dir / "report.json"
    
    if not metadata_path.exists():
        return None
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    summary = {
        'run_dir': str(run_dir),
        'timestamp': metadata.get('timestamp'),
        'parameters': metadata.get('parameters', {})
    }
    
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
        summary['summary'] = report.get('summary', {})
        summary['bankroll'] = report.get('bankroll', {})
    
    return summary
