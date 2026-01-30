import sys
import subprocess
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
from .celery_app import app

# Setup task logger
from celery.utils.log import get_task_logger
logger = get_task_logger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

def run_script(script_path, args=None, description="Script"):
    """
    Helper to run a python script as a subprocess.
    """
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    
    logger.info(f"[{description}] Starting: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=1200 # 20 mins timeout
        )
        
        if result.returncode == 0:
            logger.info(f"[{description}] Success")
            return True, result.stdout
        else:
            logger.error(f"[{description}] Failed with code {result.returncode}")
            logger.error(f"Stdout: {result.stdout}")
            logger.error(f"Stderr: {result.stderr}")
            return False, f"{result.stdout}\n{result.stderr}"
            
    except subprocess.TimeoutExpired:
        logger.error(f"[{description}] Timeout")
        return False, "Timeout"
    except Exception as e:
        logger.error(f"[{description}] Error: {e}")
        return False, str(e)

@app.task
def run_odds_pipeline_task():
    """
    Task to run the odds pipeline.
    """
    script = PROJECT_ROOT / "scripts/run_odds_pipeline.py"
    success, output = run_script(script, ["--track-lines"], "Odds Pipeline")
    if not success:
        raise Exception(f"Odds Pipeline Failed: {output}")
    return output

@app.task
def run_value_finder_task(bankroll=None, ev_threshold=None):
    """
    Task to run the value finder.
    """
    script = PROJECT_ROOT / "scripts/run_value_finder.py"
    args = ["--now", "--global-mode", "--auto"]
    
    if bankroll:
        args.extend(["--bankroll", str(bankroll)])
    if ev_threshold:
        args.extend(["--ev-threshold", str(ev_threshold)])
        
    success, output = run_script(script, args, "Value Finder")
    if not success:
        raise Exception(f"Value Finder Failed: {output}")
    return output

@app.task
def run_orchestrator_task():
    """
    Chained task: Odds -> Value.
    """
    logger.info("Starting Orchestrator Task")
    
    # 1. Run Odds
    run_odds_pipeline_task()
    
    # 2. Run Value Finder
    # We can perform logic here (e.g. only run if odds success, which is handled by raise Exception above)
    run_value_finder_task(bankroll=1000.0, ev_threshold=0.0) # Defaults
    
    # 3. Update State for Bot (/time command)
    try:
        state_file = PROJECT_ROOT / "data/scheduler_state.json"
        now = datetime.utcnow()
        next_run = now + timedelta(hours=1)
        
        state = {
            "last_run": now.isoformat(),
            "next_run": next_run.isoformat(),
            "status": "idle" 
        }
        with open(state_file, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        logger.error(f"Failed to update state file: {e}")

    logger.info("Orchestrator Task Complete")
    return "OK"
