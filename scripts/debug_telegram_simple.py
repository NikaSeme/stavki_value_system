import os
import requests
import sys
from dotenv import load_dotenv

# Load .env file
load_dotenv()

TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

print("-" * 50)
print("TELEGRAM DEBUG DIAGNOSTIC")
print("-" * 50)

# Check 1: Credentials
print(f"1. Checking Credentials:")
if not TOKEN:
    print("   [FAIL] TELEGRAM_BOT_TOKEN is missing!")
else:
    print(f"   [PASS] Token found (starts with {TOKEN[:5]}...)")

if not CHAT_ID:
    print("   [FAIL] TELEGRAM_CHAT_ID is missing!")
else:
    print(f"   [PASS] Chat ID found ({CHAT_ID})")

if not TOKEN or not CHAT_ID:
    print("\n[!] Please check your .env file.")
    sys.exit(1)

# Check 2: Connectivity
print(f"\n2. Testing API Connectivity:")
url = f"https://api.telegram.org/bot{TOKEN}/getMe"
try:
    resp = requests.get(url, timeout=10)
    data = resp.json()
    
    if resp.status_code == 200 and data.get('ok'):
        bot_name = data['result']['username']
        print(f"   [PASS] Connected to bot: @{bot_name}")
    else:
        print(f"   [FAIL] API Error: {resp.text}")
        sys.exit(1)
except Exception as e:
    print(f"   [FAIL] Connection Error: {e}")
    sys.exit(1)

# Check 3: Sending Message
print(f"\n3. Sending Test Message to {CHAT_ID}:")
send_url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
payload = {
    "chat_id": CHAT_ID,
    "text": "🔔 STAVKI DEBUG MESSAGE\n\nIf you receive this, your configuration is 100% correct.\nTime: " + os.popen('date').read().strip()
}

try:
    resp = requests.post(send_url, json=payload, timeout=10)
    data = resp.json()
    
    if resp.status_code == 200 and data.get('ok'):
        print(f"   [PASS] Message sent successfully!")
        print("\n✅ DIAGNOSTIC COMPLETE: SUCCESS")
        print("If you still don't see the message, verify you are checking the right Telegram chat.")
    else:
        print(f"   [FAIL] Sending Failed: {resp.text}")
        print("\nCommon reasons for failure:")
        print("1. You haven't started a conversation with the bot (Search for bot -> Click Start)")
        print("2. The Chat ID is from a different account/group")
except Exception as e:
    print(f"   [FAIL] Transmission Error: {e}")
