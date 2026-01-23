#!/usr/bin/env python3
"""
STAVKI Live Value Finder - One-Command Entrypoint

Finds value bets by comparing model probabilities with best available odds.
Now with comprehensive guardrails and diagnostics.

Usage:
    python run_value_finder.py --sport soccer_epl --ev-threshold 0.05
    
    python run_value_finder.py --help
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.config.env import load_env_config
from src.strategy.value_live import (
    load_latest_odds,
    select_best_prices,
    compute_no_vig_probs,
    get_model_probabilities,
    compute_ev_candidates,
    rank_value_bets,
    save_value_bets,
    diagnose_ev_outliers,
)
from src.integration.telegram_notify import send_value_alert, is_telegram_configured


def main():
    """Main entrypoint for live value finder."""
    parser = argparse.ArgumentParser(
        description="Find value bets from latest odds using model probabilities with guardrails",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_value_finder.py --sport soccer_epl --ev-threshold 0.05
  
  # With guardrails (recommended)
  python run_value_finder.py --sport soccer_epl \\
    --confirm-high-odds --outlier-drop --cap-high-odds-prob 0.20
  
  # Diagnostics mode
  python run_value_finder.py --sport soccer_epl --debug-top-k 10
  
  # With Telegram
  python run_value_finder.py --sport soccer_epl --top-n 5 --telegram
        """
    )
    
    # Basic options
    parser.add_argument(
        '--sport',
        default='soccer_epl',
        help='Sport key (must match odds filename)'
    )
    parser.add_argument(
        '--ev-threshold',
        type=float,
        default=0.08,
        help='Minimum expected value threshold (default: 0.08 = 8%%)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=10,
        help='Number of top value bets to show (default: 10)'
    )
    parser.add_argument(
        '--odds-dir',
        default='outputs/odds',
        help='Directory containing normalized odds files'
    )
    parser.add_argument(
        '--output-dir',
        default='outputs/value',
        help='Output directory for value bet results'
    )
    parser.add_argument(
        '--market',
        default='h2h',
        help='Market to analyze (default: h2h)'
    )
    
    # Strict Mode
    parser.add_argument(
        '--strict-mode',
        action='store_true',
        help='Enable strict mode: blocks baseline model, enables all guardrails, filters extreme divergence'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='ml',
        choices=['ml', 'simple', 'ensemble'],
        help='Model type: ml (trained CatBoost, default), simple (baseline 40/30/30), or ensemble (Poisson+CatBoost)'
    )
    parser.add_argument(
        '--use-neural',
        action='store_true',
        help='Include Model C (neural network) in ensemble (3-model instead of 2-model)'
    )
    parser.add_argument(
        '--allow-baseline-model',
        action='store_true',
        help='Allow baseline model in strict mode (NOT RECOMMENDED)'
    )
    parser.add_argument(
        '--max-model-market-div',
        type=float,
        help='Max model-market divergence (e.g., 0.20 = 20%%). Bets above this are filtered.'
    )
    parser.add_argument(
        '--drop-extreme-div',
        action='store_true',
        help='Drop bets with extreme divergence (>40%%) regardless of max-model-market-div'
    )
    
    # Diagnostics
    parser.add_argument(
        '--debug-top-k',
        type=int,
        help='Enable diagnostics mode and analyze top K bets'
    )
    
    # Probability validation
    parser.add_argument(
        '--prob-sum-tol',
        type=float,
        default=0.02,
        help='Probability sum tolerance (default: 0.02)'
    )
    parser.add_argument(
        '--renormalize-probs',
        action='store_true',
        help='Renormalize probabilities instead of dropping bad events'
    )
    parser.add_argument(
        '--keep-bad-prob-sum',
        action='store_true',
        help='Keep events with bad probability sums (not recommended)'
    )
    
    # Outlier detection
    parser.add_argument(
        '--outlier-drop',
        action='store_true',
        help='Drop outlier odds (>20%% gap from second-best)'
    )
    parser.add_argument(
        '--outlier-gap',
        type=float,
        default=0.20,
        help='Gap threshold for outlier detection (default: 0.20 = 20%%)'
    )
    
    # High odds guardrails
    parser.add_argument(
        '--confirm-high-odds',
        action='store_true',
        help='Require multi-bookmaker confirmation for high odds'
    )
    parser.add_argument(
        '--high-odds-threshold',
        type=float,
        default=10.0,
        help='Threshold to consider "high odds" (default: 10.0)'
    )
    parser.add_argument(
        '--high-odds-p-threshold',
        type=float,
        default=0.15,
        help='Min probability triggering high-odds check (default: 0.15)'
    )
    parser.add_argument(
        '--cap-high-odds-prob',
        type=float,
        help='Cap model probability for high odds (e.g., 0.20)'
    )
    
    # Market shrinkage
    parser.add_argument(
        '--alpha-shrink',
        type=float,
        default=1.0,
        help='Market shrinkage: p_final = alpha*p_model + (1-alpha)*p_market (default: 1.0 = no shrinkage)'
    )
    
    # Telegram
    parser.add_argument(
        '--telegram',
        action='store_true',
        help='Send Telegram alert with results'
    )
    
    args = parser.parse_args()
    
    # Apply strict-mode defaults
    if args.strict_mode:
        # Enable all guardrails by default
        if not args.confirm_high_odds:
            args.confirm_high_odds = True
        if not args.outlier_drop:
            args.outlier_drop = True
        if args.cap_high_odds_prob is None:
            args.cap_high_odds_prob = 0.15  # Cap at 15%
        if args.max_model_market_div is None:
            args.max_model_market_div = 0.20  # Max 20% divergence
        if not args.drop_extreme_div:
            args.drop_extreme_div = True
        # Ensure reasonable EV threshold
        if args.ev_threshold < 0.05:
            args.ev_threshold = 0.05
    
    # Banner
    print("=" * 70)
    print("STAVKI LIVE VALUE FINDER")
    if args.strict_mode:
        print("🔒 STRICT MODE ENABLED")
    elif args.confirm_high_odds or args.outlier_drop or args.cap_high_odds_prob:
        print("GUARDRAILS ENABLED")
    if args.debug_top_k:
        print("DIAGNOSTICS MODE")
    print("=" * 70)
    
    # Load env config
    try:
        env_config = load_env_config()
    except SystemExit:
        env_config = {}
    
    # Step 1: Load latest odds
    print(f"\n📊 Loading latest odds...")
    print(f"  Sport:        {args.sport}")
    print(f"  Odds Dir:     {args.odds_dir}")
    print(f"  Market:       {args.market}")
    
    odds_df = load_latest_odds(args.sport, args.odds_dir)
    
    if odds_df is None:
        print(f"\n❌ No odds found for sport '{args.sport}' in {args.odds_dir}")
        print(f"   Run: python run_odds_pipeline.py --sport {args.sport}")
        sys.exit(1)
    
    source_file = odds_df['_source_file'].iloc[0] if '_source_file' in odds_df.columns else 'unknown'
    print(f"  ✓ Loaded: {source_file}")
    print(f"  ✓ Total rows: {len(odds_df)}")
    
    # Step 2: Select best prices
    print(f"\n🔍 Selecting best prices...")
    if args.outlier_drop:
        print(f"  ⚙️  Outlier detection enabled (gap > {args.outlier_gap:.0%})")
    
    best_prices = select_best_prices(
        odds_df,
        check_outliers=args.outlier_drop,
        outlier_gap=args.outlier_gap
    )
    print(f"  ✓ Best prices: {len(best_prices)} outcomes")
    
    events_count = best_prices['event_id'].nunique()
    bookmakers_count = best_prices['bookmaker_title'].nunique()
    print(f"  ✓ Events: {events_count}")
    print(f"  ✓ Bookmakers: {bookmakers_count}")
    
    # Step 3: Compute no-vig probabilities
    print(f"\n🧮 Computing no-vig probabilities...")
    no_vig_probs = compute_no_vig_probs(best_prices)
    print(f"  ✓ No-vig computed for {len(no_vig_probs)} events")
    
    # Step 4: Get model probabilities
    print(f"\n🤖 Getting model probabilities...")
    
    # Initialize model based on type
    if args.model_type == 'ml':
        print(f"  Initializing ML model...")
        try:
            from src.strategy.value_live import initialize_ml_model
            initialize_ml_model()
            print(f"  ✓ ML model loaded (CatBoost + calibrator)")
        except Exception as e:
            print(f"\n❌ ERROR: Failed to load ML model")
            print(f"   {e}")
            print(f"\n   Solutions:")
            print(f"   1. Train model: python scripts/train_model.py")
            print(f"   2. Use baseline: --model-type simple")
            print(f"   3. Use ensemble: --model-type ensemble")
            sys.exit(1)
    elif args.model_type == 'ensemble':
        model_desc = "Ensemble (Poisson + CatBoost"
        if args.use_neural:
            model_desc += " + Neural)"
        else:
            model_desc += ")"
        
        print(f"  Initializing {model_desc}...")
        try:
            from src.models import EnsemblePredictor
            ensemble_predictor = EnsemblePredictor(use_neural=args.use_neural)
            
            num_models = 3 if args.use_neural else 2
            print(f"  ✓ Ensemble loaded ({num_models} models + calibration)")
        except Exception as e:
            print(f"\n❌ ERROR: Failed to load ensemble")
            print(f"   {e}")
            print(f"\n   Solutions:")
            print(f"   1. Train ensemble: python scripts/train_simple_ensemble.py")
            if args.use_neural:
                print(f"   2. Train neural model: python scripts/train_neural_model.py")
            print(f"   3. Use ML only: --model-type ml")
            sys.exit(1)
    else:
        print(f"  ⚠️  Using simple baseline model (NOT recommended for production)")
    
    # Get unique events for model prediction
    events_df = best_prices.drop_duplicates(subset=['event_id'])[
        ['event_id', 'sport_key', 'commence_time', 'home_team', 'away_team']
    ]
    
    # Get model probabilities
    if args.model_type == 'ensemble':
        # Use ensemble predictor
        probs_array, components = ensemble_predictor.predict(events_df, odds_df)
        
        # Convert to dict format
        model_probs = {}
        for i, (idx, event) in enumerate(events_df.iterrows()):
            event_id = event['event_id']
            model_probs[event_id] = {
                event['home_team']: float(probs_array[i, 0]),
                'Draw': float(probs_array[i, 1]),
                event['away_team']: float(probs_array[i, 2]),
            }
        
        print(f"  ✓ Ensemble probabilities: {len(model_probs)} events")
    else:
        # Use get_model_probabilities for ML or simple
        model_probs = get_model_probabilities(
            events_df,
            odds_df=odds_df,  # Required for ML feature extraction
            model_type=args.model_type
        )
        print(f"  ✓ Model probabilities: {len(model_probs)} events")
    
    
    # Check if baseline model and warn/block in strict mode
    from src.strategy.value_live import is_baseline_model_output
    is_baseline = is_baseline_model_output(model_probs)
    
    if is_baseline:
        if args.strict_mode and not args.allow_baseline_model:
            print(f"\n❌ ERROR: Baseline model detected - BLOCKED in strict mode")
            print(f"   Baseline model uses fixed probabilities (40/30/30)")
            print(f"   This causes unjustified EVs and is NOT suitable for real betting")
            print(f"\n   To use baseline: --model-type simple --allow-baseline-model")
            print(f"   Recommended: Use ML model (default)")
            sys.exit(1)
        elif not args.strict_mode and args.model_type == 'simple':
            print(f"\n⚠️  WARNING: Baseline model in use!")
            print(f"   • Fixed probabilities for ALL games (40/30/30)")
            print(f"   • NOT suitable for real betting - EVs may be unjustified")
            print(f"   • Use --model-type ml for production")
        elif args.allow_baseline_model:
            print(f"\n⚠️  WARNING: Baseline model explicitly allowed")
            print(f"   Strict mode filters active but model is still weak")
    elif args.model_type == 'ml':
        print(f"  ✓ Using ML model (production-ready)")
    
    # Step 5: Compute EV candidates with guardrails
    print(f"\n💰 Computing expected value...")
    print(f"  EV Threshold: {args.ev_threshold:.2%}")
    
    if args.confirm_high_odds:
        print(f"  ⚙️  High-odds confirmation: ON (threshold: {args.high_odds_threshold})")
    if args.cap_high_odds_prob:
        print(f"  ⚙️  Probability capping: {args.cap_high_odds_prob:.1%} for odds > {args.high_odds_threshold}")
    if args.alpha_shrink < 1.0:
        print(f"  ⚙️  Market shrinkage: alpha = {args.alpha_shrink}")
    if args.renormalize_probs:
        print(f"  ⚙️  Probability renormalization: ON")
    if args.max_model_market_div or args.drop_extreme_div:
        if args.max_model_market_div:
            print(f"  ⚙️  Model-market divergence filter: Max {args.max_model_market_div:.0%}")
        if args.drop_extreme_div:
            print(f"  ⚙️  Extreme divergence drop: ON (>40%)")
    
    candidates = compute_ev_candidates(
        model_probs,
        best_prices,
        all_odds_df=odds_df if args.confirm_high_odds else None,
        threshold=args.ev_threshold,
        market_key=args.market,
        prob_sum_tol=args.prob_sum_tol,
        drop_bad_prob_sum=not args.keep_bad_prob_sum,
        renormalize=args.renormalize_probs,
        confirm_high_odds=args.confirm_high_odds,
        high_odds_threshold=args.high_odds_threshold,
        high_odds_p_threshold=args.high_odds_p_threshold,
        cap_high_odds_prob=args.cap_high_odds_prob,
        alpha_shrink=args.alpha_shrink,
        max_model_market_div=args.max_model_market_div,
        drop_extreme_div=args.drop_extreme_div,
    )
    
    print(f"  ✓ Value bets found: {len(candidates)}")
    
    if not candidates:
        print(f"\n⚠️  No value bets found above {args.ev_threshold:.2%} threshold")
        print(f"   Try lowering threshold with: --ev-threshold 0.03")
        print(f"   Or disable guardrails to see raw output")
        sys.exit(0)
    
    # Step 6: Diagnostics (if enabled)
    if args.debug_top_k:
        print(f"\n🔬 Running diagnostics...")
        diag_path = diagnose_ev_outliers(
            candidates,
            odds_df,
            model_probs,
            no_vig_probs,
            top_k=args.debug_top_k
        )
        print(f"  ✓ Diagnostics saved: {diag_path}")
        print(f"\n💡 Review diagnostics to identify issues:")
        print(f"   cat {diag_path}")
    
    # Step 7: Rank and limit
    top_bets = rank_value_bets(candidates, top_n=args.top_n)
    
    # Step 8: Save outputs
    print(f"\n💾 Saving results...")
    csv_file, json_file = save_value_bets(top_bets, args.sport, args.output_dir)
    print(f"  ✓ CSV:  {csv_file}")
    print(f"  ✓ JSON: {json_file}")
    
    # Step 9: Display summary
    print(f"\n" + "=" * 70)
    print("TOP VALUE BETS")
    print("=" * 70)
    
    for i, bet in enumerate(top_bets[:args.top_n], 1):
        print(f"\n{i}. {bet['selection']} @ {bet['odds']} ({bet['bookmaker']})")
        print(f"   {bet['home_team']} vs {bet['away_team']}")
        
        if 'p_final' in bet and bet['p_final'] != bet['p_model']:
            print(f"   EV: +{bet['ev_pct']:.2f}% | Model: {bet['p_model']*100:.1f}% | Final: {bet['p_final']*100:.1f}% | Implied: {bet['p_implied']*100:.1f}%")
        else:
            print(f"   EV: +{bet['ev_pct']:.2f}% | Model: {bet['p_model']*100:.1f}% | Implied: {bet['p_implied']*100:.1f}%")
        
        print(f"   Starts: {bet['commence_time']}")
    
    # Summary stats
    avg_ev = sum(b['ev'] for b in top_bets) / len(top_bets)
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total value bets:     {len(candidates)}")
    print(f"Showing top:          {min(args.top_n, len(top_bets))}") 
    print(f"Average EV:           +{avg_ev*100:.2f}%")
    print(f"Best EV:              +{top_bets[0]['ev_pct']:.2f}%")
    print(f"EV range:             +{top_bets[-1]['ev_pct']:.2f}% to +{top_bets[0]['ev_pct']:.2f}%")
    
    # Step 10: Optional Telegram alert
    if args.telegram:
        print(f"\n📱 Sending Telegram alert...")
        success = send_value_alert(top_bets, top_n=5)
        if not success and not is_telegram_configured():
            print(f"   Configure with: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
    else:
        if is_telegram_configured():
            print(f"\n💡 Tip: Add --telegram to send alerts to Telegram")
    
    print("=" * 70)
    print(f"\n✅ Value finder complete!")
    print(f"📁 Output directory: {Path(args.output_dir).absolute()}")
    
    # Recommendations
    if not (args.confirm_high_odds or args.outlier_drop):
        print(f"\n💡 Recommended guardrails:")
        print(f"   --confirm-high-odds --outlier-drop")


if __name__ == "__main__":
    main()
