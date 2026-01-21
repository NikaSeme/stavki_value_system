"""
Pipeline package for STAVKI betting system.
End-to-end integration of all components.
"""

from .evaluation import calculate_metrics, generate_evaluation_report, load_results
from .reports import collect_warnings, generate_report, save_report_json, save_report_txt
from .run_archive import (
    create_run_directory,
    get_run_summary,
    list_runs,
    save_run_artifacts,
    save_run_metadata,
    setup_run_logging,
)
from .run_pipeline import run_pipeline

__all__ = [
    "run_pipeline",
    "generate_report",
    "save_report_json",
    "save_report_txt",
    "collect_warnings",
    "create_run_directory",
    "save_run_metadata",
    "save_run_artifacts",
    "setup_run_logging",
    "list_runs",
    "get_run_summary",
    "load_results",
    "calculate_metrics",
    "generate_evaluation_report",
]
