import pandas as pd
import json
import os
import sys

# Constants
DATA_FILE = 'data/processed/epl_features_2021_2024.csv'
OUTPUT_DIR = 'outputs/audit_v2/A3_leakage'
REQUIRED_COLS = ['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'Season']
TIME_COLS = ['Date'] # Add others if found

os.makedirs(OUTPUT_DIR, exist_ok=True)

report = {
    "file": DATA_FILE,
    "status": "FAIL",
    "missing_columns": [],
    "time_columns_found": [],
    "date_format_check": "FAIL"
}

try:
    if not os.path.exists(DATA_FILE):
        report["error"] = "File not found"
        print(f"File not found: {DATA_FILE}")
    else:
        df = pd.read_csv(DATA_FILE)
        
        # Check Columns
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        report["missing_columns"] = missing
        
        if missing:
            print(f"Missing columns: {missing}")
        else:
            report["status"] = "PASS"
        
        # Check Time Columns
        found_time = [c for c in TIME_COLS if c in df.columns]
        report["time_columns_found"] = found_time
        
        # Check Date Format
        if 'Date' in df.columns:
            try:
                pd.to_datetime(df['Date'])
                report["date_format_check"] = "PASS"
            except:
                report["date_format_check"] = "FAIL"
                
except Exception as e:
    report["error"] = str(e)

# Save Report
with open(os.path.join(OUTPUT_DIR, 'schema_check_report.json'), 'w') as f:
    json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))

if report["status"] != "PASS":
    sys.exit(1)
