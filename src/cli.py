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
    
    # TODO: Implement analysis logic
    logger.warning("Analysis module not yet implemented (placeholder)")
    logger.info("This will be implemented in future tasks:")
    logger.info("  - T010: Data ingestion")
    logger.info("  - T020: Feature engineering")
    logger.info("  - T030-T040: Models A & B")
    logger.info("  - T050: Ensemble & calibration")


@cli.command()
@click.option("--start-date", help="Backtest start date (YYYY-MM-DD)")
@click.option("--end-date", help="Backtest end date (YYYY-MM-DD)")
@click.pass_context
def backtest(ctx: click.Context, start_date: str, end_date: str) -> None:
    """
    Run backtest on historical data.
    
    Simulates betting strategy on past matches to evaluate performance.
    """
    logger = ctx.obj["logger"]
    
    logger.info("Starting backtest...")
    if start_date and end_date:
        logger.info(f"Period: {start_date} to {end_date}")
    
    # TODO: Implement backtest
    logger.warning("Backtest module not yet implemented (placeholder)")


@cli.command()
@click.pass_context
def monitor(ctx: click.Context) -> None:
    """
    Monitor live matches and send notifications for value bets.
    
    Runs continuously, checking for betting opportunities.
    """
    logger = ctx.obj["logger"]
    cfg: Config = ctx.obj["config"]
    
    logger.info("Starting live monitoring...")
    logger.info("Press Ctrl+C to stop")
    
    if cfg.dry_run:
        logger.warning("Running in DRY RUN mode - no bets will be placed")
    
    # TODO: Implement monitoring
    logger.warning("Monitor module not yet implemented (placeholder)")


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
    "--features",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to features CSV file"
)
@click.option(
    "--odds",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to odds CSV file"
)
@click.option(
    "--bankroll",
    type=float,
    default=1000.0,
    help="Total bankroll"
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
    "--output",
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
    kelly_fraction: float,
    max_stake_pct: float,
    output: Path
) -> None:
    """
    Run end-to-end betting pipeline.
    
    Loads features and odds, generates predictions, calculates EV and stakes,
    and saves recommendations.
    """
    import pandas as pd
    from .pipeline.run_pipeline import run_pipeline, save_recommendations
    
    logger = ctx.obj['logger']
    
    logger.info("═" * 60)
    logger.info("STAVKI End-to-End Pipeline")
    logger.info("═" * 60)
    logger.info(f"Features: {features}")
    logger.info(f"Odds: {odds}")
    logger.info(f"Bankroll: ${bankroll:,.2f}")
    logger.info(f"Kelly Fraction: {kelly_fraction}")
    logger.info(f"Max Stake: {max_stake_pct}%")
    logger.info("═" * 60)
    
    try:
        # Load data
        logger.info("Loading data...")
        features_df = pd.read_csv(features)
        odds_df = pd.read_csv(odds)
        
        logger.info(f"✓ Loaded {len(features_df)} features")
        logger.info(f"✓ Loaded {len(odds_df)} odds")
        
        # Run pipeline
        recommendations = run_pipeline(
            features_df,
            odds_df,
            bankroll=bankroll,
            kelly_fraction=kelly_fraction,
            max_stake_fraction=max_stake_pct / 100.0
        )
        
        if len(recommendations) == 0:
            logger.warning("⚠ No recommendations generated (no positive EV bets)")
            return
        
        # Save results
        paths = save_recommendations(recommendations, output)
        
        # Display summary
        logger.info("")
        logger.info("═" * 60)
        logger.info("RECOMMENDATIONS SUMMARY")
        logger.info("═" * 60)
        logger.info(f"Total bets: {len(recommendations)}")
        logger.info(f"Total stake: ${recommendations['stake'].sum():,.2f}")
        logger.info(f"Average EV: {recommendations['ev'].mean():.2%}")
        logger.info(f"Average odds: {recommendations['odds'].mean():.2f}")
        logger.info("═" * 60)
        logger.info(f"✓ Saved to: {paths['csv']}")
        logger.info(f"✓ Saved to: {paths['json']}")
        logger.info("═" * 60)
        
        click.echo(f"\n✓ Pipeline complete! {len(recommendations)} recommendations generated.")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
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
