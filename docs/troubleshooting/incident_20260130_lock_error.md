# Systems Analysis: Scheduler Lock Error
**Date:** 2026-01-30
**Component:** `scripts/run_scheduler.py`
**Error:** `❌ Error: Another instance of the scheduler is already running.`

## 1. Problem Diagnosis
The logs indicate repeated failures to acquire the application lock.
Code inspection confirms usage of `fcntl.flock(LOCK_EX | LOCK_NB)`. 
Unlike simple file-existence locks, `fcntl` locks are tied to an open file descriptor held by an active process.

**Implication:**
- The lock is NOT "stale" in the file-system sense (OS releases fcntl locks if process dies).
- **A process is actively running** and holding the lock.
- **Hypothesis:** The user likely executed `./scripts/start_all.sh` previously, spawning a background process (`nohup ... &`), which is still alive. The Systemd service is contending with this manual instance.

## 2. Proposed Solution (The "Clean Slate")
To restore order, we must ensure **all** scheduling processes are terminated before letting Systemd take over.

**Steps:**
1. **Terminate Ghosts:** `sudo pkill -f run_scheduler.py`
2. **Cleanup:** `sudo rm -f /tmp/stavki_scheduler.lock` (Just in case)
3. **Restart:** `sudo systemctl restart stavki-scheduler`

## 3. Prevention
- Users should rely *solely* on Systemd in production (`start_all.sh` is for dev/manual testing).
- Future `start_all.sh` updates should check for Systemd service status before running.
