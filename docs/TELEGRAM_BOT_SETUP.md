# Telegram Bot Setup - Complete Guide

## Overview

This guide will set up a Telegram bot to receive alerts from your betting system.

## Step 1: Create Telegram Bot with BotFather

1. **Open Telegram** on your phone or desktop
2. **Search for @BotFather** (official Telegram bot creator)
3. **Start a chat** with BotFather
4. **Send command:** `/newbot`
5. **Choose a name** for your bot (e.g., "Stavki Value Bot")
6. **Choose a username** for your bot (must end in 'bot', e.g., "stavki_value_bot")
7. **Save the API token** BotFather gives you - looks like:
   ```
   123456789:ABCdefGHIjklMNOpqrsTUVwxyz
   ```

## Step 2: Get Your Chat ID

### Method 1: Using @userinfobot
1. Search for **@userinfobot** in Telegram
2. Start a chat with it
3. It will send you your **Chat ID** (a number like `123456789`)
4. Save this number

### Method 2: Using API call (after creating bot)
1. Send any message to your bot (the one you just created)
2. Open browser and go to:
   ```
   https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
   ```
   Replace `<YOUR_BOT_TOKEN>` with the token from Step 1
3. Look for `"chat":{"id":123456789}` in the response
4. Save this chat ID

## Step 3: Configure Environment Variables

1. **Open your `.env` file** in the project root:
   ```bash
   nano /Users/macuser/Documents/something/stavki_value_system/.env
   ```

2. **Add these lines** (replace with your actual values):
   ```bash
   # Telegram Bot Configuration
   TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
   TELEGRAM_CHAT_ID=123456789
   
   # Optional: Encryption key for account credentials
   BETTING_ACCOUNTS_KEY=your_encryption_key_here
   ```

3. **Save the file** (Ctrl+X, then Y, then Enter in nano)

## Step 4: Verify .env is in .gitignore

1. **Check `.gitignore` contains `.env`:**
   ```bash
   cat .gitignore | grep .env
   ```
   
2. **If not found, add it:**
   ```bash
   echo ".env" >> .gitignore
   ```

This prevents accidentally committing your secrets to Git.

## Step 5: Test Telegram Bot Connection

1. **Run the test script:**
   ```bash
   cd /Users/macuser/Documents/something/stavki_value_system
   source venv/bin/activate
   python src/alerts/telegram_bot.py
   ```

2. **Expected output:**
   ```
   ✓ Connected to bot: @your_bot_username
   ✓ Test alert sent - check Telegram
   ```

3. **Check your Telegram** - you should receive a test message!

## Step 6: Send Your First Alert

**Test the alert manager:**
```bash
python src/alerts/alert_manager.py
```

This will send 4 test alerts:
- Value bet alert
- Line movement alert  
- Performance alert
- System health alert

## Step 7: Integrate with Your System

The bot is now ready to use in your betting system. Alerts will be sent automatically when:
- Value bets are detected (EV > 5%)
- Sharp line movements occur (>10% in <12h)
- Performance warnings trigger (drawdown >20%)
- System errors occur

## Troubleshooting

### "Connection failed" error
- Check your bot token is correct
- Verify bot token has no extra spaces
- Make sure you started a chat with your bot first

### No messages received
- Verify chat ID is correct
- Check you messaged the bot at least once
- Try the @userinfobot method to confirm your chat ID

### "TELEGRAM_BOT_TOKEN not found"
- Make sure `.env` file exists in project root
- Check the token line has no typos
- Restart your terminal/reload environment variables

## Security Notes

✅ **DO:**
- Keep `.env` file in `.gitignore`
- Never share your bot token
- Revoke token if accidentally exposed (via @BotFather)

❌ **DON'T:**
- Commit `.env` to Git
- Share screenshots containing tokens
- Use the same bot for multiple projects (security best practice)

## Advanced: Custom Alert Messages

Edit `src/alerts/alert_manager.py` to customize alert formats:
- Change emojis
- Add/remove information
- Adjust thresholds
- Filter by league

---

**Setup Complete!** Your Telegram bot is ready to send betting alerts.
