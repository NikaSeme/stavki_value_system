"""
Telegram bot for STAVKI betting system.

Features:
- Run betting pipeline
- Get recommendations
- View statistics
- System status
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from ..logging_setup import get_logger
from ..pipeline.evaluation import calculate_metrics, load_results
from ..pipeline.reports import collect_warnings, generate_report
from ..pipeline.run_pipeline import run_pipeline

logger = get_logger(__name__)


class StavkiBot:
    """Telegram bot for STAVKI betting system."""
    
    def __init__(self, token: str, allowed_users: list):
        """
        Initialize bot.
        
        Args:
            token: Telegram bot token
            allowed_users: List of allowed user IDs
        """
        self.token = token
        self.allowed_users = allowed_users
        self.app = Application.builder().token(token).build()
        
        # Register command handlers
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("help", self.help_command))
        self.app.add_handler(CommandHandler("run", self.run_betting_pipeline))
        self.app.add_handler(CommandHandler("status", self.status))
        self.app.add_handler(CommandHandler("stats", self.stats))
        self.app.add_handler(CommandHandler("latest", self.latest_run))
        
        logger.info(f"Bot initialized with {len(allowed_users)} allowed users")
    
    def check_auth(self, user_id: int) -> bool:
        """Check if user is authorized."""
        is_auth = user_id in self.allowed_users
        if not is_auth:
            logger.warning(f"Unauthorized access attempt from user {user_id}")
        return is_auth
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Send welcome message."""
        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"
        
        logger.info(f"Start command from user {user_id} ({username})")
        
        if not self.check_auth(user_id):
            await update.message.reply_text("⛔ Unauthorized. Contact admin.")
            return
        
        welcome_msg = (
            "🎯 *STAVKI Betting System Bot*\n\n"
            "Welcome to your automated betting assistant!\n\n"
            "*Available commands:*\n"
            "/run - Run betting pipeline\n"
            "/latest - View latest recommendations\n"
            "/status - Check system status\n"
            "/stats - View performance statistics\n"
            "/help - Show detailed help\n\n"
            "💡 Tip: Use `/run 1000 0.10` to set bankroll and EV threshold"
        )
        
        await update.message.reply_text(welcome_msg, parse_mode='Markdown')
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show help."""
        if not self.check_auth(update.effective_user.id):
            return
        
        help_msg = (
            "📖 *STAVKI Bot Help*\n\n"
            "*Commands:*\n\n"
            "`/run` - Run pipeline with defaults\n"
            "  • Bankroll: $1000\n"
            "  • EV Threshold: 10%\n"
            "  • Max Bets: 5\n\n"
            "`/run <bankroll> <ev>` - Custom parameters\n"
            "  • Example: `/run 500 0.15`\n"
            "  • Bankroll: $500, EV: 15%\n\n"
            "`/latest` - View last recommendations\n\n"
            "`/status` - System health check\n\n"
            "`/stats` - Performance metrics\n"
            "  • ROI, Hit Rate, Profit\n\n"
            "💡 *Tips:*\n"
            "• Higher EV threshold = fewer but safer bets\n"
            "• Lower threshold = more bets, higher risk\n"
            "• Recommended: 8-15% EV threshold"
        )
        
        await update.message.reply_text(help_msg, parse_mode='Markdown')
    
    async def run_betting_pipeline(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Run betting pipeline."""
        user_id = update.effective_user.id
        
        if not self.check_auth(user_id):
            await update.message.reply_text("⛔ Unauthorized")
            return
        
        # Parse parameters
        bankroll = 1000.0
        ev_threshold = 0.10
        max_bets = 5
        
        if context.args:
            try:
                bankroll = float(context.args[0])
                if len(context.args) > 1:
                    ev_threshold = float(context.args[1])
                if len(context.args) > 2:
                    max_bets = int(context.args[2])
            except ValueError:
                await update.message.reply_text(
                    "❌ Invalid parameters\n"
                    "Usage: `/run <bankroll> <ev_threshold> <max_bets>`\n"
                    "Example: `/run 1000 0.10 5`",
                    parse_mode='Markdown'
                )
                return
        
        await update.message.reply_text(
            f"🔄 *Running Pipeline...*\n\n"
            f"💰 Bankroll: ${bankroll:,.2f}\n"
            f"📊 EV Threshold: {ev_threshold:.1%}\n"
            f"🎯 Max Bets: {max_bets}",
            parse_mode='Markdown'
        )
        
        try:
            # Load data
            data_dir = Path("data/processed")
            features_df = pd.read_csv(data_dir / "features.csv")
            odds_df = pd.read_csv(data_dir / "odds.csv")
            
            logger.info(f"Running pipeline: bankroll={bankroll}, ev={ev_threshold}")
            
            # Run pipeline
            recommendations = run_pipeline(
                features_df,
                odds_df,
                bankroll=bankroll,
                kelly_fraction=0.5,
                max_stake_fraction=0.05,
                ev_threshold=ev_threshold
            )
            
            # Limit bets
            if len(recommendations) > max_bets:
                recommendations = recommendations.nlargest(max_bets, 'ev')
            
            # Collect warnings
            warnings = collect_warnings(features_df, odds_df, recommendations, bankroll)
            
            # Generate report
            report = generate_report(recommendations, bankroll, ev_threshold, warnings)
            
            # Send results
            if len(recommendations) == 0:
                await update.message.reply_text(
                    "⚠️ *No Bets Found*\n\n"
                    f"No bets meet the EV threshold of {ev_threshold:.1%}\n"
                    "Try lowering the threshold.",
                    parse_mode='Markdown'
                )
            else:
                # Summary message
                summary_msg = (
                    f"✅ *Pipeline Complete!*\n\n"
                    f"🎯 *Recommendations:* {len(recommendations)} bets\n"
                    f"💵 *Total Stake:* ${report['summary']['total_stake']:,.2f}\n"
                    f"📊 *Avg EV:* {report['summary']['avg_ev']:.2%}\n"
                    f"💰 *Bankroll Used:* {report['bankroll']['utilization_pct']:.1f}%\n"
                    f"🎁 *Potential Profit:* ${report['summary']['total_potential_profit']:,.2f}\n\n"
                )
                
                # Add warnings
                if warnings:
                    summary_msg += "⚠️ *Warnings:*\n"
                    for warning in warnings[:3]:  # Limit to 3
                        summary_msg += f"• {warning}\n"
                    summary_msg += "\n"
                
                await update.message.reply_text(summary_msg, parse_mode='Markdown')
                
                # Send top 3 bets details
                for i, bet in recommendations.head(3).iterrows():
                    bet_msg = (
                        f"📌 *Bet #{i+1}*\n\n"
                        f"⚽ *Match:* {bet.get('home_team', 'N/A')} vs {bet.get('away_team', 'N/A')}\n"
                        f"📅 *Date:* {bet.get('date', 'N/A')}\n"
                        f"🎯 *Outcome:* {bet['outcome'].upper()}\n"
                        f"📊 *Probability:* {bet['probability']:.1%}\n"
                        f"💰 *Odds:* {bet['odds']:.2f}\n"
                        f"📈 *EV:* {bet['ev']:.2%}\n"
                        f"💵 *Stake:* ${bet['stake']:.2f}\n"
                        f"🎁 *Potential:* ${bet['potential_profit']:.2f}"
                    )
                    await update.message.reply_text(bet_msg, parse_mode='Markdown')
                
                # Save to file
                output_dir = Path("outputs/telegram_runs")
                output_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                recommendations.to_csv(output_dir / f"bets_{timestamp}.csv", index=False)
                
                logger.info(f"Pipeline complete: {len(recommendations)} bets generated")
            
        except FileNotFoundError as e:
            error_msg = (
                "❌ *Data Not Found*\n\n"
                "Missing required files:\n"
                "• `data/processed/features.csv`\n"
                "• `data/processed/odds.csv`\n\n"
                "Please ensure data files exist."
            )
            await update.message.reply_text(error_msg, parse_mode='Markdown')
            logger.error(f"Pipeline error: {e}")
            
        except Exception as e:
            await update.message.reply_text(
                f"❌ *Error*\n\n{str(e)}",
                parse_mode='Markdown'
            )
            logger.error(f"Pipeline failed: {e}", exc_info=True)
    
    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """System status."""
        if not self.check_auth(update.effective_user.id):
            return
        
        # Check data files
        data_dir = Path("data/processed")
        features_exists = (data_dir / "features.csv").exists()
        odds_exists = (data_dir / "odds.csv").exists()
        
        status_msg = (
            "🔍 *System Status*\n\n"
            f"{'🟢' if features_exists else '🔴'} Features Data\n"
            f"{'🟢' if odds_exists else '🔴'} Odds Data\n"
            "🟢 Pipeline Ready\n"
            "🟢 Bot Connected\n\n"
            f"📅 Status Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        await update.message.reply_text(status_msg, parse_mode='Markdown')
    
    async def stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Performance statistics."""
        if not self.check_auth(update.effective_user.id):
            return
        
        try:
            # Try to load results
            results_path = Path("tests/fixtures/results.csv")
            
            if results_path.exists():
                results_df = load_results(results_path)
                metrics = calculate_metrics(results_df)
                
                stats_msg = (
                    "📊 *Performance Statistics*\n\n"
                    f"🎯 Total Bets: {metrics['number_of_bets']}\n"
                    f"✅ Wins: {metrics['wins']} ({metrics['hit_rate']:.1f}%)\n"
                    f"❌ Losses: {metrics['losses']}\n"
                    f"💰 Profit: ${metrics['profit']:,.2f}\n"
                    f"📈 ROI: {metrics['roi']:.2f}%\n"
                    f"💵 Avg Stake: ${metrics['avg_stake']:.2f}\n"
                    f"📊 Avg Odds: {metrics['avg_odds']:.2f}"
                )
            else:
                stats_msg = (
                    "📊 *Performance Statistics*\n\n"
                    "No betting history available yet.\n"
                    "Results will appear after placing bets."
                )
            
            await update.message.reply_text(stats_msg, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(
                f"❌ Error loading stats: {str(e)}",
                parse_mode='Markdown'
            )
    
    async def latest_run(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show latest run results."""
        if not self.check_auth(update.effective_user.id):
            return
        
        try:
            output_dir = Path("outputs/telegram_runs")
            
            if not output_dir.exists():
                await update.message.reply_text(
                    "📭 No runs yet. Use `/run` to generate recommendations.",
                    parse_mode='Markdown'
                )
                return
            
            # Find latest file
            csv_files = list(output_dir.glob("bets_*.csv"))
            
            if not csv_files:
                await update.message.reply_text(
                    "📭 No runs yet. Use `/run` to generate recommendations.",
                    parse_mode='Markdown'
                )
                return
            
            latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
            recommendations = pd.read_csv(latest_file)
            
            timestamp = latest_file.stem.replace('bets_', '')
            
            summary_msg = (
                f"📋 *Latest Run* ({timestamp})\n\n"
                f"🎯 Bets: {len(recommendations)}\n"
                f"💵 Total Stake: ${recommendations['stake'].sum():,.2f}\n"
                f"📊 Avg EV: {recommendations['ev'].mean():.2%}\n\n"
                "Top 3 bets:"
            )
            
            await update.message.reply_text(summary_msg, parse_mode='Markdown')
            
            # Send top 3
            for i, bet in recommendations.head(3).iterrows():
                bet_msg = (
                    f"📌 *#{i+1}*\n"
                    f"{bet.get('home_team', 'N/A')} vs {bet.get('away_team', 'N/A')}\n"
                    f"🎯 {bet['outcome'].upper()} @ {bet['odds']:.2f}\n"
                    f"💵 ${bet['stake']:.2f} (EV: {bet['ev']:.1%})"
                )
                await update.message.reply_text(bet_msg, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(
                f"❌ Error: {str(e)}",
                parse_mode='Markdown'
            )
    
    def run(self):
        """Start the bot."""
        logger.info("Starting STAVKI Telegram Bot...")
        logger.info(f"Allowed users: {self.allowed_users}")
        
        print("🤖 STAVKI Telegram Bot Starting...")
        print(f"✓ Allowed users: {self.allowed_users}")
        print("✓ Commands registered")
        print("🔄 Polling for updates...\n")
        
        self.app.run_polling(allowed_updates=Update.ALL_TYPES)
