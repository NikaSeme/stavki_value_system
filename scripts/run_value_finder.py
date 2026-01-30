#!/usr/bin/env python3
"""
STAVKI Live Value Finder (V5 Production)

Main entrypoint for finding and alerting on value bets.
Supports scheduled runs, immediate checks, and strict production guardrails.

Usage:
    python run_value_finder.py --now --telegram        # Run immediately & alert
    python run_value_finder.py --top 10               # View top 10 bets
    python run_value_finder.py --dry-run              # Simulate full run
    python run_value_finder.py --help
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import sys
import os
import csv
import json
import glob
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.env import load_env_config
from src.strategy.value_live import (
    load_latest_odds,
    select_best_prices,
    compute_no_vig_probs, # Still imported but not used in new main logic
    get_model_probabilities,
    compute_ev_candidates,
    rank_value_bets, # Still imported but not used in new main logic
    save_value_bets, # Still imported but not used in new main logic
    diagnose_ev_outliers, # Still imported but not used in new main logic
    initialize_ml_model
)
from src.integration.telegram_notify import send_value_alert, is_telegram_configured, send_custom_message

# V5 Policies
# V5 Policies: Defining 'Major' vs 'Small' leagues
# Major = Best in country or elite international competitions
MAJOR_LEAGUES = [
    'soccer_epl', 'soccer_spain_la_liga', 'soccer_italy_serie_a',
    'soccer_germany_bundesliga', 'soccer_france_ligue_one',
    'soccer_uefa_champions_league', 'soccer_uefa_europa_league',
    'soccer_fifa_world_cup', 'soccer_uefa_european_championship',
    'basketball_euroleague'
]

MAX_SMALL_LEAGUE_GAMES = 3

def load_leagues_config():
    """Load leagues configuration from YAML."""
    import yaml
    config_path = Path(__file__).parent.parent / 'config' / 'leagues.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def is_league_trained(sport_key):
    """Check if a league has training data available."""
    try:
        config = load_leagues_config()
        
        # Check soccer leagues
        for league in config.get('soccer', []):
            if league['key'] == sport_key:
                return league.get('training_data', False)
        
        # Check basketball leagues  
        for league in config.get('basketball', []):
            if league['key'] == sport_key:
                return league.get('training_data', False)
        
        # Unknown league = no training data
        return False
    except Exception as e:
        print(f"⚠️ Warning: Could not load leagues config: {e}")
        # Fallback: Only EPL is trained
        return sport_key == 'soccer_epl'

def get_git_revision_short_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('ascii')
    except:
        return "unknown"

# -----------------------------------------------------------------------------
# INTERACTIVE HELPERS (M20)
# -----------------------------------------------------------------------------
def is_interactive_run(args):
    """Determine if we should run in interactive mode."""
    # Explicit non-interactive flags
    if args.auto or args.non_interactive:
        return False
    
    # Explicit interactive flag
    if args.interactive:
        return True
        
    # Default: Interactive if TTY and not automated env var
    return sys.stdin.isatty() and os.environ.get("STAVKI_AUTOMATED") != "1"

def prompt_float(label, default=None, min_val=None, max_val=None):
    """Prompt user for float input with validation."""
    while True:
        try:
            default_str = f" [{default}]" if default is not None else ""
            user_input = input(f"{label}{default_str}: ").strip()
            
            if not user_input and default is not None:
                return default
            
            val = float(user_input)
            
            if min_val is not None and val < min_val:
                print(f"  ❌ Value must be >= {min_val}")
                continue
            
            if max_val is not None and val > max_val:
                print(f"  ❌ Value must be <= {max_val}")
                continue
                
            return val
        except ValueError:
            print("  ❌ Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(0)

def show_menu():
    """Show main menu and return choice."""
    print("\n" + "="*40)
    print(" 🎮 STAVKI MANUAL RUN MENU")
    print("="*40)
    print(" 1. 🚀 Fetch Odds + Run Value Finder")
    print(" 2. ⚡ Run Value Finder (Latest Odds)")
    print(" 3. 📝 View Top Bets (No Alerts)")
    print(" 4. 🧪 Dry Run (Simulation)")
    print(" 5. 🚪 Exit")
    print("-" * 40)
    
    while True:
        choice = input("Select option [1-5]: ").strip()
        if choice in ['1', '2', '3', '4', '5']:
            return choice
        print("Invalid choice.")

def main():
    parser = argparse.ArgumentParser(
        description="Stavki Value Finder V5 (Production)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Standard Flags
    parser.add_argument('--now', action='store_true', help='Run immediately')
    parser.add_argument('--top', type=int, help='View top N bets')
    parser.add_argument('--dry-run', action='store_true', help='Simulate run')
    parser.add_argument('--telegram', action='store_true', help='Send Telegram alerts')
    parser.add_argument('--global-mode', action='store_true', default=True, help='Scan all active leagues')
    parser.add_argument('--leagues', type=str, help='Comma-separated list of league keys to include')
    
    # Interactive / Auto flags (M20)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--interactive', action='store_true', help='Force interactive mode')
    group.add_argument('--non-interactive', action='store_true', help='Force non-interactive mode')
    group.add_argument('--auto', action='store_true', help='Alias for --non-interactive')
    
    # Load persistent settings
    settings_path = Path("config/user_settings.json")
    persistent_settings = {}
    if settings_path.exists():
        try:
            with open(settings_path) as f:
                persistent_settings = json.load(f)
        except Exception:
            pass

    # Tuning params
    default_br = persistent_settings.get("bankroll", 40.0)
    default_ev = persistent_settings.get("ev_threshold", 0.08)
    
    parser.add_argument('--bankroll', type=float, default=default_br, help=f'Bankroll override (default: {default_br})')
    parser.add_argument('--ev-threshold', type=float, default=default_ev, help=f'EV threshold override (default: {default_ev})')
    parser.add_argument('--ev-max', type=float, default=10.0, help='Maximum EV threshold (e.g. 0.5 for 50%)')
    parser.add_argument('--target-avg-ev', type=float, help='Target Average EV %%')
    
    # Advanced / Debug
    parser.add_argument('--odds-dir', default='outputs/odds')
    parser.add_argument('--output-dir', default='outputs/value')
    parser.add_argument('--debug-top-k', type=int, help='Run diagnostics on top K bets')
    
    args = parser.parse_args()
    
    # Determine mode
    interactive = is_interactive_run(args)
    
    if interactive:
        choice = show_menu()
        
        if choice == '1':
            print("\n🔄 Running Odds Pipeline first...")
            try:
                subprocess.run(["python3", "scripts/run_odds_pipeline.py", "--track-lines"], check=True)
            except subprocess.CalledProcessError:
                print("❌ Odds fetch failed. Aborting.")
                sys.exit(1)
            args.now = True
            args.telegram = True 
        elif choice == '2':
            args.now = True
            args.telegram = True
        elif choice == '3':
            args.top = 10
            args.telegram = False
            args.now = True
        elif choice == '4':
            args.dry_run = True
            args.now = True
            args.telegram = False
        elif choice == '5':
            print("Bye!")
            sys.exit(0)
            
        print("\n💰 CAPITAL CONFIGURATION")
        if args.bankroll:
            print(f"  Bankroll: {args.bankroll} (from flag)")
        else:
            args.bankroll = prompt_float("  Enter Bankroll ($)", default=40.0, min_val=1.0)
            

    else:
        # Non-interactive validation
        if args.bankroll and args.bankroll <= 0:
            print("Error: Bankroll must be positive")
            sys.exit(1)
    
    # Logging setup occurs after parsing
    log_dir = Path("audit_pack/RUN_LOGS")

    # Handle --top shortcut
    if args.top:
        print(f"👀 View Mode: Showing Top {args.top} Bets")
        args.telegram = False # Disable alerts in view mode

    print("="*60)
    print(f"STAVKI V5 PRODUCTION RUN ({datetime.now(timezone.utc)} UTC)")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'LIVE'}")
    print("="*60)

    # Load Environment
    load_env_config()

    # 1. Load Data
    print(f"\n📊 Loading Odds Data...")
    import pandas as pd
    if args.global_mode:
        search_pattern = f"{args.odds_dir}/events_latest_*.csv"
        files = sorted(glob.glob(search_pattern))
        if not files:
            print(f"❌ No odds files found in {args.odds_dir}")
            sys.exit(1)
        latest_file = files[-1]
        print(f"  ✓ Unified File: {latest_file}")
        odds_df = pd.read_csv(latest_file)
    else:
        odds_df = load_latest_odds(args.sport, args.odds_dir)

    if odds_df is None or odds_df.empty:
        print("❌ Data Load Failed")
        sys.exit(1)

    # 2. Select Best Prices
    print(f"\n🔍 Selecting Market Best...")
    best_prices = select_best_prices(odds_df, check_outliers=False)
    
    # Selective League Filter (M65)
    if getattr(args, 'leagues', None):
        target_leagues = [x.strip() for x in args.leagues.split(',')]
        print(f"  🎯 Filtering for leagues: {target_leagues}")
        best_prices = best_prices[best_prices['sport_key'].isin(target_leagues)]

    events_unique = best_prices[['event_id', 'sport_key', 'home_team', 'away_team', 'commence_time']].drop_duplicates('event_id')
    print(f"  ✓ {len(events_unique)} Events across {best_prices['sport_key'].nunique()} leagues")

    # 3. Initialize Models
    print(f"\n⚙️  Initializing Models...")
    try:
        initialize_ml_model() # Loads CatBoost for Soccer
    except Exception as e:
        print(f"❌ Model Load Error: {e}")
        sys.exit(1)

    from src.strategy.blending import get_blending_alpha # Import locally to avoid circle if any
    
    # 4. Global Prediction Loop
    all_candidates = []

    for sport_key, sport_events in events_unique.groupby('sport_key'):
        # Training Data Check (v6.5) - Only predict leagues with training data
        if not is_league_trained(sport_key):
            print(f"\n⏭️  Skipping {sport_key} (no training data available)")
            continue

        # Get Dynamic Alpha (V6 Smart Blending)
        alpha = get_blending_alpha(sport_key, aggressive=True)
        print(f"\n👉 {sport_key} ({len(sport_events)} events) | Alpha: {alpha:.2f}")

        # Predict & EV
        try:
            model_probs, s_data = get_model_probabilities(sport_events, odds_df[odds_df['sport_key']==sport_key], model_type='ml')
            
            # Use dynamic alpha for V6
            candidates = compute_ev_candidates(
                model_probs,
                best_prices[best_prices['sport_key'] == sport_key],
                sentiment_data=s_data,
                threshold=args.ev_threshold,
                market_key='h2h',
                prob_sum_tol=0.01,
                drop_bad_prob_sum=True,
                renormalize=False,
                confirm_high_odds=True,
                high_odds_threshold=10.0,
                high_odds_p_threshold=0.15,
                cap_high_odds_prob=0.15, 
                alpha_shrink=alpha, # V6 Dynamic Alpha
                max_model_market_div=None, 
                drop_extreme_div=False,
                bankroll=args.bankroll if args.bankroll else 1000.0
            )
            for c in candidates: c['sport_key'] = sport_key
            
            # Application of Max EV (M68)
            if getattr(args, 'ev_max', None):
                candidates = [c for c in candidates if c['ev'] <= args.ev_max]
                
            all_candidates.extend(candidates)
            
            # M50: Trace Logging (Show analysis even if no value)
            print(f"   📊 Analysis Trace (Found {len(model_probs)} matches):")
            for i, (eid, probs) in enumerate(list(model_probs.items())[:3]): # Trace first 3
                event = sport_events[sport_events['event_id'] == eid].iloc[0]
                home = event['home_team']
                away = event['away_team']
                p_str = " | ".join([f"{k}: {v:.1%}" for k, v in probs.items()])
                print(f"      - {home} vs {away}: {p_str}")
            
            print(f"   ✓ {len(candidates)} value candidates found.")

        except Exception as e:
            print(f"   ❌ Error processing {sport_key}: {e}")
            continue

    # 5. Filtering & Policies
    print(f"\n🛡 Applying V5 Policies...")

    # Sort by EV
    all_candidates.sort(key=lambda x: x['ev'], reverse=True)
    
    # Safety EV Cap (v6.5): Prevent extreme bets due to out-of-distribution calibration
    # Until multi-league training is complete, cap EVs at 50%
    EV_SAFETY_CAP = 50.0  # Maximum allowed EV (%)
    pre_cap_count = len(all_candidates)
    all_candidates = [c for c in all_candidates if c['ev_pct'] <= EV_SAFETY_CAP]
    capped_count = pre_cap_count - len(all_candidates)
    if capped_count > 0:
        print(f"  🛡️ Safety Cap: Filtered {capped_count} bets with EV > {EV_SAFETY_CAP}%")

    # Log Raw Predictions (Audit)
    # Log Raw Predictions (Audit)
    audit_dir = Path("audit_pack/A9_live")
    audit_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.dry_run and all_candidates:
        # Better: use fixed path
        p_file = audit_dir / "predictions.csv"

        # Simple Append
        file_exists = p_file.exists()
        with open(p_file, 'a', newline='') as f:
            fieldnames = ['timestamp'] + list(all_candidates[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists: writer.writeheader()
            ts = datetime.now(timezone.utc).isoformat()
            for c in all_candidates:
                row = c.copy()
                row['timestamp'] = ts
                writer.writerow(row)

    filtered_bets = []
    minor_league_bets = []

    # V5 Policy: Minor League Cap (Max 5% of TOTAL alerts)
    # This acts on the FINAL set. Simple approach:
    # 1. Separate Major vs Minor
    # 2. Pick all Major > Threshold
    # 3. Pick Minor up to Cap

    # First pass: Standard Filters (High EV, Limits, Dedupe done later)
    # We apply the 'per league' limits here first

    # Global counters
    league_counts = {}
    match_counts = {}

    outliers = []
    confirmation = []

    for c in all_candidates:
        # Sanity Tags (No longer blocking)
        if c['odds'] > 10.0:
            c['reason'] = 'HIGH_ODDS'
            outliers.append(c)
        if c['ev_pct'] > 35.0:
            c['reason'] = 'EV > 35%'
            confirmation.append(c)

        # Diversification
        lg = c['sport_key']
        eid = c['event_id']

        # No limits per league or match


        league_counts[lg] = league_counts.get(lg, 0) + 1
        match_counts[eid] = match_counts.get(eid, 0) + 1

        # Classification
        if lg in MAJOR_LEAGUES:
            filtered_bets.append(c)
        else:
            minor_league_bets.append(c)

    # Sort Minor League bets by EV and apply global cap
    minor_league_bets.sort(key=lambda x: x['ev'], reverse=True)
    final_minor = minor_league_bets[:MAX_SMALL_LEAGUE_GAMES]
    
    # Combine Major and Capped Minor
    final_bets = filtered_bets + final_minor
    


    final_bets.sort(key=lambda x: x['ev'], reverse=True)

    # Limit to Top N for Display/Alert
    top_n_display = args.top if args.top else 20 # Default 20 for alerts (v6.0)

    print(f"  ✓ Candidates: {len(all_candidates)}")
    print(f"  ✓ Highly Divergent: {len(outliers) + len(confirmation)} (Warning only)")
    print(f"  ✓ Major Bets: {len(filtered_bets)}")
    print(f"  ✓ Minor Bets: {len(final_minor)}")
    print(f"  ✓ Final Selection: {len(final_bets)}")

    # Log Diverted
    if not args.dry_run:
        if confirmation:
            # Save to needs_confirmation
            log_csv(audit_dir / "needs_confirmation.csv", confirmation)
        if outliers:
            log_csv(audit_dir / "outliers.csv", outliers)

    # 6. Output & Alert
    if final_bets:
        # Save Top Bets for Bot (Overwrite latest)
        top_file = audit_dir / "top_ev_bets.csv"
        # Always write top bets regardless of dry-run if we want /top to work in debug
        # But instructions say "Ensure CSV is written even when --telegram is not used, unless --dry-run is set."
        # Actually, usually --dry-run might imply we don't want to affect state. 
        # But for /top visibility, maybe we do? 
        # Let's check instructions: "Ensure CSV is written even when --telegram is not used, unless --dry-run is set."
        if not args.dry_run:
            try:
                with open(top_file, 'w', newline='') as f:
                    fieldnames = ['home_team', 'away_team', 'selection', 'odds', 'bookmaker', 'quality_score', 'ev_pct', 'stake_pct', 'sport_key', 'event_id']
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                    writer.writeheader()
                    writer.writerows(final_bets)
                print(f"  ✓ Saved top bets to {top_file}")
            except Exception as e:
                print(f"  ⚠️ Could not save top bets CSV: {e}")

        print(f"\n" + "="*60)
        print(f"TOP {min(len(final_bets), top_n_display)} BETS (Full list in CSV)")
        print("="*60)
        for i, b in enumerate(final_bets[:top_n_display], 1):
             print(f"{i}. {b['selection']} @ {b['odds']} ({b['home_team']} vs {b['away_team']}) EV: {b['ev_pct']}%")

        # Build Stamp
        build_data = {
            'commit': get_git_revision_short_hash(),
            'timestamp': datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            'model_version': 'v3.2' # From metadata
        }

        if args.telegram:
            print(f"\n📱 Sending Telegram...")
            send_value_alert(
                final_bets, 
                top_n=top_n_display, 
                build_data=build_data, 
                csv_path=top_file, 
                dry_run=args.dry_run
            )

            # Log Sents (if not dry run)
            if not args.dry_run:
                log_csv(audit_dir / "alerts_sent.csv", final_bets)
                print(f"  ✓ Logged to alerts_sent.csv")

        # M10: Send Report Flag
        if getattr(args, 'send_report', False):
            print(f"\n📊 Generating Summary Report...")
            report_lines = [
                "📊 *STAVKI RUN SUMMARY*",
                f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
                f"Total Candidates: {len(all_candidates)}",
                f"Final Selection: {len(final_bets)}",
                f"Avg EV: {sum(b['ev_pct'] for b in final_bets)/len(final_bets):.2f}%" if final_bets else "Avg EV: 0%",
                f"Status: {'DRY-RUN' if args.dry_run else 'LIVE'}"
            ]
            report_msg = "\n".join(report_lines)
            if send_custom_message(report_msg):
                print("  ✓ Summary report sent to Telegram.")
            else:
                print("  ⚠️ Failed to send summary report.")

    else:
        msg = f"\n💤 *No Value Bets Found*\nScanned `{len(all_candidates)}` candidates.\nThreshold: `{args.ev_threshold*100}%` EV."
        print(msg)
        if args.telegram:
            send_custom_message(msg)

def log_csv(path, rows):
    if not rows: return
    file_exists = path.exists()
    with open(path, 'a', newline='') as f:
        fieldnames = ['timestamp'] + list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists: writer.writeheader()
        ts = datetime.now(timezone.utc).isoformat()
        for r in rows:
            row = r.copy()
            row['timestamp'] = ts
            writer.writerow(row)

if __name__ == "__main__":
    main()
