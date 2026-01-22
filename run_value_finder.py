#!/usr/bin/env python3
"""
STAVKI Live Value Finder - One-Command Entrypoint

Finds value bets by comparing model probabilities with best available odds.

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
)
from src.integration.telegram_notify import send_value_alert, is_telegram_configured


def main():
    """Main entrypoint for live value finder."""
    parser = argparse.ArgumentParser(
        description="Find value bets from latest odds using model probabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_value_finder.py --sport soccer_epl --ev-threshold 0.05
  python run_value_finder.py --sport basketball_nba --top-n 5 --telegram
  python run_value_finder.py --odds-dir myodds/ --output-dir myvalue/
        """
    )
    
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
        '--telegram',
        action='store_true',
        help='Send Telegram alert with results'
    )
    parser.add_argument(
        '--market',
        default='h2h',
        help='Market to analyze (default: h2h)'
    )
    
    args = parser.parse_args()
    
    # Banner
    print("=" * 70)
    print("STAVKI LIVE VALUE FINDER")
    print("=" * 70)
    
    # Load env config (for potential future use)
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
    best_prices = select_best_prices(odds_df)
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
    print(f"  ⚠️  Using simple baseline model (future: ensemble integration)")
    
    # Get unique events for model prediction
    events_df = best_prices.drop_duplicates(subset=['event_id'])[
        ['event_id', 'sport_key', 'commence_time', 'home_team', 'away_team']
    ]
    
    model_probs = get_model_probabilities(events_df, model_type='simple')
    print(f"  ✓ Model probabilities: {len(model_probs)} events")
    
    # Step 5: Compute EV candidates
    print(f"\n💰 Computing expected value...")
    print(f"  EV Threshold: {args.ev_threshold:.2%}")
    
    candidates = compute_ev_candidates(
        model_probs,
        best_prices,
        threshold=args.ev_threshold,
        market_key=args.market
    )
    
    print(f"  ✓ Value bets found: {len(candidates)}")
    
    if not candidates:
        print(f"\n⚠️  No value bets found above {args.ev_threshold:.2%} threshold")
        print(f"   Try lowering threshold with: --ev-threshold 0.03")
        sys.exit(0)
    
    # Step 6: Rank and limit
    top_bets = rank_value_bets(candidates, top_n=args.top_n)
    
    # Step 7: Save outputs
    print(f"\n💾 Saving results...")
    csv_file, json_file = save_value_bets(top_bets, args.sport, args.output_dir)
    print(f"  ✓ CSV:  {csv_file}")
    print(f"  ✓ JSON: {json_file}")
    
    # Step 8: Display summary
    print(f"\n" + "=" * 70)
    print("TOP VALUE BETS")
    print("=" * 70)
    
    for i, bet in enumerate(top_bets[:args.top_n], 1):
        print(f"\n{i}. {bet['selection']} @ {bet['odds']} ({bet['bookmaker']})")
        print(f"   {bet['home_team']} vs {bet['away_team']}")
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
    
    # Step 9: Optional Telegram alert
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


if __name__ == "__main__":
    main()
