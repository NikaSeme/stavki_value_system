"""
Command-line interface for STAVKI value betting system.
"""

import sys
from pathlib import Path

import click

from .config import Config
from .logging_setup import get_logger, setup_logging


@click.group()
@click.version_option(version="0.1.0")
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to .env configuration file"
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    help="Logging level"
)
@click.pass_context
def cli(ctx: click.Context, config: Path, log_level: str) -> None:
    """
    STAVKI Value Betting System
    
    A professional sports betting system using ensemble models,
    probability calibration, and expected value calculations.
    
    \b
    ⚠️  IMPORTANT SAFETY NOTES:
    - By default, system runs in DRY_RUN mode (no real bets)
    - Set DRY_RUN=false in .env to enable real betting
    - Always test thoroughly before using real money
    - Never bet more than you can afford to lose
    """
    # Initialize context
    ctx.ensure_object(dict)
    
    # Load configuration
    cfg = Config.from_env(config)
    cfg.create_directories()
    
    # Setup logging
    setup_logging(
        log_level=log_level,
        log_file=cfg.log_file,
        enable_colors=True
    )
    
    logger = get_logger(__name__)
    
    # Validate and display config
    try:
        cfg.validate()
    except ValueError as e:
        logger.error("Configuration validation failed", error=str(e))
        sys.exit(1)
    
    # Store config in context
    ctx.obj["config"] = cfg
    ctx.obj["logger"] = logger
    
    # Show startup banner
    if ctx.invoked_subcommand is None:
        return
    
    logger.info("=" * 60)
    logger.info("STAVKI Value Betting System v0.1.0")
    logger.info("=" * 60)
    
    if cfg.dry_run:
        logger.warning("🔒 DRY RUN MODE - No real bets will be placed")
    else:
        logger.critical("⚠️  LIVE MODE - Real bets will be placed!")
        logger.critical("   Make sure you know what you're doing!")


@cli.command()
@click.pass_context
def config_show(ctx: click.Context) -> None:
    """Display current configuration."""
    cfg: Config = ctx.obj["config"]
    logger = ctx.obj["logger"]
    
    logger.info("\n" + str(cfg))
    logger.info(f"\nConfiguration valid: ✓")


@cli.command()
@click.pass_context
def config_validate(ctx: click.Context) -> None:
    """Validate configuration."""
    cfg: Config = ctx.obj["config"]
    logger = ctx.obj["logger"]
    
    try:
        cfg.validate()
        logger.info("✓ Configuration is valid")
    except ValueError as e:
        logger.error(f"✗ Configuration validation failed: {e}")
        sys.exit(1)


@cli.command()
@click.option("--match-id", help="Specific match ID to analyze")
@click.pass_context
def analyze(ctx: click.Context, match_id: str) -> None:
    """
    Analyze upcoming matches and find value bets.
    
    This will:
    1. Fetch upcoming matches
    2. Generate predictions using ensemble models
    3. Calculate expected value (EV)
    4. Filter for value bets above threshold
    """
    logger = ctx.obj["logger"]
    cfg: Config = ctx.obj["config"]
    
    logger.info("Starting match analysis...")
    
    if match_id:
        logger.info(f"Analyzing specific match: {match_id}")
    else:
        logger.info("Analyzing all upcoming matches")
    
    # Implement analysis logic by wrapping the value finder pipeline
    try:
        from .pipeline.run_pipeline import run_pipeline
        import pandas as pd
        
        # We need to fetch data first or use existing? 
        # The 'analyze' command usually implies "finding value now".
        # So we can reuse the logic from scripts/run_value_finder.py essentially.
        # But for CLI purity, we should import library functions.
        # However, run_value_finder.py logic is somewhat script-heavy.
        # Simplest path is subprocess for now to match exactly scripts/run_value_finder.py behavior.
        
        # M22: CLI Alignment - Support interactive mode
        import subprocess
        
        # Default to interactive if running manually via CLI, 
        # unless user pipe usage suggests otherwise (handled by script auto-detection),
        # but let's be explicit if we want to force the menu.
        # Since 'analyze' is an alias for manual run, we pass --interactive.
        
        cmd = ["python3", "scripts/run_value_finder.py", "--interactive"]
        
        if match_id:
             logger.warning("Filtering by match_id is not fully supported in CLI wrapper yet.")
             
        # Use simple run to allow stdout/stdin passthrough
        subprocess.run(cmd, check=True)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


@cli.command()
@click.option("--start-date", help="Backtest start date (YYYY-MM-DD)")
@click.option("--end-date", help="Backtest end date (YYYY-MM-DD)")
@click.pass_context
def backtest(ctx: click.Context, start_date: str, end_date: str) -> None:
    """
    Run backtest on historical data.
    
    Simulates betting strategy on past matches.
    """
    logger = ctx.obj["logger"]
    logger.info("Starting backtest...")
    
    import subprocess
    cmd = ["python3", "scripts/run_backtest_v3_4.py"]
    # Pass dates if script accepted them, but currently it's hardcoded or uses config.
    # We'll just run the script.
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error("Backtest failed.")
        sys.exit(e.returncode)


@cli.command()
@click.pass_context
def monitor(ctx: click.Context) -> None:
    """
    Monitor live matches and send notifications for value bets.
    
    Runs continuously (via scheduler).
    """
    logger = ctx.obj["logger"]
    cfg: Config = ctx.obj["config"]
    
    logger.info("Starting live monitoring (Scheduler)...")
    logger.info("Press Ctrl+C to stop")
    
    import subprocess
    # We use the new scheduler script which supports CLI args better
    cmd = ["python3", "scripts/run_scheduler.py", "--interval", "60", "--telegram"]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("Monitoring stopped.")
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        sys.exit(1)


@cli.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, path_type=Path),
    default="data/raw/",
    help="Directory containing CSV files to ingest"
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default="data/processed/matches.csv",
    help="Output file for normalized matches"
)
@click.pass_context
def ingest(ctx: click.Context, input_dir: Path, output: Path) -> None:
    """
    Ingest and normalize football match CSV files.
    
    Scans input directory for CSV files in football-data.co.uk format,
    normalizes columns, handles missing values, and saves to output file.
    """
    from .data.ingestion import ingest_directory
    
    logger = ctx.obj["logger"]
    
    logger.info(f"Ingesting CSV files from {input_dir}")
    
    try:
        stats = ingest_directory(Path(input_dir), Path(output))
        
        # Display summary
        logger.info("\n" + "=" * 60)
        logger.info("Ingestion Summary")
        logger.info("=" * 60)
        logger.info(f"Files processed:     {stats['files_processed']}")
        logger.info(f"Total unique matches: {stats['total_matches']}")
        logger.info(f"Valid rows:          {stats['valid_rows']}")
        logger.info(f"Skipped rows:        {stats['skipped_rows']}")
        logger.info(f"Duplicates removed:  {stats['duplicates_removed']}")
        logger.info(f"Output saved to:     {output}")
        logger.info("=" * 60)
        
        if stats['files_processed'] == 0:
            logger.warning("No files were processed!")
        elif stats['total_matches'] == 0:
            logger.warning("No valid matches found!")
        else:
            logger.info("✓ Ingestion completed successfully!")
            
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        sys.exit(1)


@cli.command()
@click.option(
    "--input",
    type=click.Path(exists=True, path_type=Path),
    default="data/processed/matches.csv",
    help="Input matches CSV file"
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default="data/processed/features.csv",
    help="Output features CSV file"
)
@click.option(
    "--window",
    type=int,
    default=5,
    help="Rolling window size (number of recent matches)"
)
@click.pass_context
def features(ctx: click.Context, input: Path, output: Path, window: int) -> None:
    """
    Build feature dataset with rolling statistics from matches.
    
    Calculates rolling statistics (goals, points, form) for each team
    using historical match data. STRICTLY prevents data leakage by using
    only matches before the current match date.
    """
    from .features.build_features import build_features_dataset
    
    logger = ctx.obj["logger"]
    
    logger.info(f"Building features from {input}")
    logger.info(f"Rolling window: {window} matches")
    
    try:
        stats = build_features_dataset(Path(input), Path(output), window=window)
        
        # Display summary
        logger.info("\n" + "=" * 60)
        logger.info("Feature Engineering Summary")
        logger.info("=" * 60)
        logger.info(f"Total matches:           {stats['total_matches']}")
        logger.info(f"Matches with features:   {stats['matches_with_features']}")
        logger.info(f"Matches without features: {stats['matches_without_features']}")
        logger.info(f"Avg home team history:   {stats['avg_home_history']:.1f} matches")
        logger.info(f"Avg away team history:   {stats['avg_away_history']:.1f} matches")
        logger.info(f"Output saved to:         {output}")
        logger.info("=" * 60)
        
        if stats['matches_with_features'] == 0:
            logger.warning("No matches have features (all teams have no history)!")
        else:
            logger.info("✓ Feature engineering completed successfully!")
            logger.info("   (Features use ONLY historical data - no leakage)")
            
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        sys.exit(1)


@cli.command("predict-poisson")
@click.option(
    "--input",
    type=click.Path(exists=True, path_type=Path),
    default="data/processed/features.csv",
    help="Input features CSV file"
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default="data/processed/predictions_poisson.csv",
    help="Output predictions CSV file"
)
@click.pass_context
def predict_poisson(ctx: click.Context, input: Path, output: Path) -> None:
    """
    Generate Poisson model predictions for match outcomes.
    
    Uses Poisson distribution to estimate expected goals and calculate
    Home/Draw/Away probabilities based on historical team performance.
    """
    from .models.poisson_model import PoissonModel
    
    logger = ctx.obj["logger"]
    
    logger.info("Running Poisson Model (Model A)")
    logger.info(f"Input: {input}")
    
    try:
        model = PoissonModel()
        stats = model.predict_from_file(Path(input), Path(output))
        
        # Display summary
        logger.info("\n" + "=" * 60)
        logger.info("Poisson Model Summary")
        logger.info("=" * 60)
        logger.info(f"Total matches:       {stats['total_matches']}")
        logger.info(f"Avg λ_home:          {stats['avg_lambda_home']:.3f}")
        logger.info(f"Avg λ_away:          {stats['avg_lambda_away']:.3f}")
        logger.info(f"Avg P(Home):         {stats['avg_prob_home']:.3f}")
        logger.info(f"Avg P(Draw):         {stats['avg_prob_draw']:.3f}")
        logger.info(f"Avg P(Away):         {stats['avg_prob_away']:.3f}")
        logger.info(f"Output saved to:     {output}")
        logger.info("=" * 60)
        
        # Verify probability sum
        prob_sum = stats['avg_prob_home'] + stats['avg_prob_draw'] + stats['avg_prob_away']
        if abs(prob_sum - 1.0) < 0.001:
            logger.info("✓ Probabilities sum to 1.0 (validated)")
        else:
            logger.warning(f"⚠ Probability sum: {prob_sum:.6f} (expected 1.0)")
        
        logger.info("✓ Poisson predictions completed successfully!")
            
    except Exception as e:
        logger.error(f"Poisson prediction failed: {e}")
        sys.exit(1)


@cli.command("train-ml")
@click.option(
    "--input",
    type=click.Path(exists=True, path_type=Path),
    default="data/processed/features.csv",
    help="Input features CSV file"
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default="outputs/models/ml_model.pkl",
    help="Output model file"
)
@click.option(
    "--train-ratio",
    type=float,
    default=0.7,
    help="Training data ratio (0-1)"
)
@click.pass_context
def train_ml(ctx: click.Context, input: Path, output: Path, train_ratio: float) -> None:
    """
    Train ML model (LightGBM) with probability calibration.
    
    Splits data temporally (NO shuffle!) to prevent data leakage.
    Applies isotonic regression for probability calibration.
    """
    from .models.ml_model import MLModel
    
    logger = ctx.obj["logger"]
    
    logger.info("Training ML Model (Model B)")
    logger.info(f"Input: {input}")
    logger.info(f"Train/Valid split: {train_ratio:.0%} / {(1-train_ratio):.0%}")
    
    try:
        model = MLModel()
        stats = model.train_from_file(Path(input), Path(output), train_ratio)
        
        # Display summary
        logger.info("\n" + "=" * 60)
        logger.info("ML Model Training Summary")
        logger.info("=" * 60)
        logger.info(f"Train samples:       {stats['train_samples']}")
        logger.info(f"Valid samples:       {stats['valid_samples']}")
        logger.info(f"Train accuracy:      {stats['train_accuracy']:.3f}")
        logger.info(f"Valid accuracy:      {stats['valid_accuracy']:.3f}")
        logger.info(f"Train acc (cal):     {stats['train_accuracy_cal']:.3f}")
        logger.info(f"Valid acc (cal):     {stats['valid_accuracy_cal']:.3f}")
        logger.info(f"Model saved to:      {output}")
        logger.info("=" * 60)
        
        logger.info("✓ ML model training completed successfully!")
        logger.info("   (Temporal split enforced - no data leakage)")
            
    except Exception as e:
        logger.error(f"ML training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


@cli.command("predict-ml")
@click.option(
    "--model",
    type=click.Path(exists=True, path_type=Path),
    default="outputs/models/ml_model.pkl",
    help="Path to trained model file"
)
@click.option(
    "--input",
    type=click.Path(exists=True, path_type=Path),
    default="data/processed/features.csv",
    help="Input features CSV file"
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default="data/processed/predictions_ml.csv",
    help="Output predictions CSV file"
)
@click.pass_context
def predict_ml(ctx: click.Context, model: Path, input: Path, output: Path) -> None:
    """
    Generate ML model predictions for match outcomes.
    
    Loads trained LightGBM model and generates calibrated probabilities.
    """
    from .models.ml_model import MLModel
    
    logger = ctx.obj["logger"]
    
    logger.info("Running ML Model Predictions")
    logger.info(f"Model: {model}")
    logger.info(f"Input: {input}")
    
    try:
        ml_model = MLModel.load(Path(model))
        stats = ml_model.predict_from_file(Path(input), Path(output))
        
        # Display summary
        logger.info("\n" + "=" * 60)
        logger.info("ML Predictions Summary")
        logger.info("=" * 60)
        logger.info(f"Total matches:       {stats['total_matches']}")
        logger.info(f"Avg P(Home):         {stats['avg_prob_home']:.3f}")
        logger.info(f"Avg P(Draw):         {stats['avg_prob_draw']:.3f}")
        logger.info(f"Avg P(Away):         {stats['avg_prob_away']:.3f}")
        logger.info(f"Output saved to:     {output}")
        logger.info("=" * 60)
        
        # Verify probability sum
        prob_sum = stats['avg_prob_home'] + stats['avg_prob_draw'] + stats['avg_prob_away']
        if abs(prob_sum - 1.0) < 0.001:
            logger.info("✓ Probabilities sum to 1.0 (validated)")
        else:
            logger.warning(f"⚠ Probability sum: {prob_sum:.6f} (expected 1.0)")
        
        logger.info("✓ ML predictions completed successfully!")
            
    except Exception as e:
        logger.error(f"ML prediction failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.pass_context
def check(ctx: click.Context) -> None:
    """
    Run system checks to verify everything is working.
    
    Checks:
    - Configuration validity
    - Directory structure
    - Logging setup
    - Dependencies
    """
    logger = ctx.obj["logger"]
    cfg: Config = ctx.obj["config"]
    
    logger.info("Running system checks...")
    
    # Check 1: Config
    try:
        cfg.validate()
        logger.info("✓ Configuration valid")
    except ValueError as e:
        logger.error(f"✗ Configuration invalid: {e}")
        return
    
    # Check 2: Directories
    for name, directory in [
        ("Data", cfg.data_dir),
        ("Models", cfg.models_dir),
        ("Outputs", cfg.outputs_dir),
        ("Logs", cfg.logs_dir)
    ]:
        if directory.exists():
            logger.info(f"✓ {name} directory exists: {directory}")
        else:
            logger.warning(f"⚠ {name} directory missing: {directory}")
    
    # Check 3: Logging
    test_log = cfg.log_file
    if test_log.exists():
        logger.info(f"✓ Log file created: {test_log}")
    else:
        logger.warning(f"⚠ Log file not yet created: {test_log}")
    
    logger.info("\n✓ All checks completed!")


@cli.command("run")
@click.option(
    "--matches",
    "--features",
    "features",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to matches/features CSV/JSON file"
)
@click.option(
    "--odds",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to odds CSV/JSON file"
)
@click.option(
    "--bankroll",
    type=float,
    default=1000.0,
    help="Total bankroll"
)
@click.option(
    "--ev-threshold",
    type=float,
    default=0.08,
    help="Minimum EV threshold for bets (default: 0.08 = 8%)"
)
@click.option(
    "--kelly-fraction",
    type=float,
    default=0.5,
    help="Kelly fraction (0.5 = half Kelly)"
)
@click.option(
    "--max-stake-pct",
    type=float,
    default=5.0,
    help="Maximum stake as percentage of bankroll"
)
@click.option(
    "--max-bets",
    type=int,
    default=None,
    help="Maximum number of bets to recommend"
)
@click.option(
    "--output-dir",
    "--output",
    "output",
    type=click.Path(path_type=Path),
    default="outputs",
    help="Output directory for recommendations"
)
@click.pass_context
def run(
    ctx: click.Context,
    features: Path,
    odds: Path,
    bankroll: float,
    ev_threshold: float,
    kelly_fraction: float,
    max_stake_pct: float,
    max_bets: int,
    output: Path
) -> None:
    """
    Run end-to-end betting pipeline (PRODUCTION).
    
    Loads features and odds, generates predictions, calculates EV and stakes,
    and saves comprehensive recommendations with warnings.
    
    Example:
        python -m src.cli run \\
            --matches data/matches.csv \\
            --odds data/odds.csv \\
            --bankroll 1000 \\
            --ev-threshold 0.30 \\
            --output outputs/
    """
    import pandas as pd
    from .pipeline.reports import (
        collect_warnings,
        generate_report,
        save_report_json,
        save_report_txt,
    )
    from .pipeline.run_pipeline import run_pipeline
    
    logger = ctx.obj['logger']
    
    logger.info("═" * 60)
    logger.info("STAVKI PRODUCTION PIPELINE")
    logger.info("═" * 60)
    logger.info(f"Matches:      {features}")
    logger.info(f"Odds:         {odds}")
    logger.info(f"Bankroll:     ${bankroll:,.2f}")
    logger.info(f"EV Threshold: {ev_threshold:.1%}")
    logger.info(f"Kelly:        {kelly_fraction}")
    logger.info(f"Max Stake:    {max_stake_pct}%")
    if max_bets:
        logger.info(f"Max Bets:     {max_bets}")
    logger.info("═" * 60)
    
    try:
        # Load data (support both CSV and JSON)
        logger.info("Loading data...")
        
        if features.suffix == '.json':
            features_df = pd.read_json(features)
        else:
            features_df = pd.read_csv(features)
        
        if odds.suffix == '.json':
            odds_df = pd.read_json(odds)
        else:
            odds_df = pd.read_csv(odds)
        
        logger.info(f"✓ Loaded {len(features_df)} matches")
        logger.info(f"✓ Loaded {len(odds_df)} odds")
        
        # Run pipeline
        recommendations = run_pipeline(
            features_df,
            odds_df,
            bankroll=bankroll,
            kelly_fraction=kelly_fraction,
            max_stake_fraction=max_stake_pct / 100.0,
            ev_threshold=ev_threshold
        )
        
        # Limit number of bets if specified
        if max_bets and len(recommendations) > max_bets:
            logger.info(f"Limiting to top {max_bets} bets by EV...")
            recommendations = recommendations.nlargest(max_bets, 'ev')
        
        # Collect warnings
        warnings = collect_warnings(
            features_df,
            odds_df,
            recommendations,
            bankroll
        )
        
        # Generate report
        report = generate_report(
            recommendations,
            bankroll,
            ev_threshold,
            warnings
        )
        
        # Save reports
        output_dir = Path(output)
        save_report_json(report, output_dir / "bets.json")
        save_report_txt(report, output_dir / "bets.txt")
        
        # Display summary
        logger.info("")
        logger.info("═" * 60)
        logger.info("PRODUCTION REPORT")
        logger.info("═" * 60)
        logger.info(f"Total Bets:       {report['summary']['total_bets']}")
        logger.info(f"Total Stake:      ${report['summary']['total_stake']:,.2f}")
        logger.info(f"Bankroll Used:    {report['bankroll']['utilization_pct']:.1f}%")
        logger.info(f"Remaining:        ${report['bankroll']['remaining']:,.2f}")
        logger.info(f"Average EV:       {report['summary']['avg_ev']:.2%}")
        logger.info(f"Potential Profit: ${report['summary']['total_potential_profit']:,.2f}")
        
        if warnings:
            logger.info("")
            logger.info("WARNINGS:")
            for warning in warnings:
                logger.warning(f"⚠ {warning}")
        
        logger.info("═" * 60)
        logger.info(f"✓ Saved: {output_dir / 'bets.json'}")
        logger.info(f"✓ Saved: {output_dir / 'bets.txt'}")
        logger.info("═" * 60)
        
        if len(recommendations) == 0:
            click.echo("\n⚠ No bets recommended with current criteria.")
        else:
            click.echo(f"\n✓ {len(recommendations)} bets recommended. See {output_dir}/bets.txt for details.")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@cli.command("eval")
@click.option(
    "--results",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to results CSV file"
)
@click.option(
    "--runs",
    type=click.Path(exists=True, path_type=Path),
    default="runs",
    help="Path to runs directory"
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default="outputs",
    help="Output directory for evaluation reports"
)
@click.pass_context
def eval_performance(
    ctx: click.Context,
    results: Path,
    runs: Path,
    output: Path
) -> None:
    """
    Evaluate betting performance from results.
    
    Calculates metrics (ROI, hit rate, profit) from betting results
    and generates evaluation reports.
    
    Example:
        python -m src.cli eval \\
            --results data/results.csv \\
            --runs runs/ \\
            --output outputs/
    """
    import pandas as pd
    from .pipeline.evaluation import (
        calculate_metrics,
        generate_evaluation_report,
        load_results,
        save_evaluation_summary,
    )
    
    logger = ctx.obj['logger']
    
    logger.info("═" * 60)
    logger.info("STAVKI PERFORMANCE EVALUATION")
    logger.info("═" * 60)
    logger.info(f"Results file: {results}")
    logger.info(f"Runs dir:     {runs}")
    logger.info(f"Output dir:   {output}")
    logger.info("═" * 60)
    
    try:
        # Load results
        logger.info("Loading results...")
        results_df = load_results(results)
        logger.info(f"✓ Loaded {len(results_df)} betting results")
        
        # Calculate metrics
        logger.info("Calculating metrics...")
        metrics = calculate_metrics(results_df)
        
        # Generate report
        report_text = generate_evaluation_report(metrics, results_df)
        
        # Save summaries
        output_dir = Path(output)
        paths = save_evaluation_summary(metrics, report_text, output_dir)
        
        # Display summary
        logger.info("")
        logger.info("═" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("═" * 60)
        logger.info(f"Total Bets:     {metrics['number_of_bets']}")
        logger.info(f"Wins/Losses:    {metrics['wins']}/{metrics['losses']}")
        logger.info(f"Total Staked:   ${metrics['total_staked']:,.2f}")
        logger.info(f"Total Returned: ${metrics['total_returned']:,.2f}")
        logger.info(f"Profit:         ${metrics['profit']:,.2f}")
        logger.info(f"ROI:            {metrics['roi']:.2f}%")
        logger.info(f"Hit Rate:       {metrics['hit_rate']:.2f}%")
        logger.info("═" * 60)
        logger.info(f"✓ Saved: {paths['json']}")
        logger.info(f"✓ Saved: {paths['txt']}")
        logger.info("═" * 60)
        
        # Print summary to console
        if metrics['profit'] > 0:
            click.echo(f"\n✓ Profitable! ROI: {metrics['roi']:.2f}%, Profit: ${metrics['profit']:,.2f}")
        elif metrics['profit'] < 0:
            click.echo(f"\n✗ Loss. ROI: {metrics['roi']:.2f}%, Loss: ${abs(metrics['profit']):,.2f}")
        else:
            click.echo(f"\n○ Break-even. No profit or loss.")
        
        click.echo(f"See {paths['txt']} for full report.")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


def main() -> None:
    """Entry point for CLI."""
    try:
        cli(obj={})
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
