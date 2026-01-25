# Cleanup Log

## Legacy Files Removed
- `T*.json` (Legacy tracking files)
- `PLAN 1.txt` (Old plan)
- `README_ODDS_INTEGRATION.md` (Duplicate docs)
- `stavki_audit_pack*.zip` (Old audit packs)
- `outputs/audit_v2` (Previous audit v2 output)
- `outputs/audit_pack_v2`
- `outputs/diagnostics`

## Moves
- Moved `run_scheduler.py` -> `scripts/run_scheduler.py`
- Moved `run_odds_pipeline.py` -> `scripts/run_odds_pipeline.py`
- Moved `run_value_finder.py` -> `scripts/run_value_finder.py`

## New Structure
- `/src`: Core application code
- `/scripts`: Execution and utility scripts
- `/config`: Configuration files
- `/audit_pack`: Generated audit evidence (A0-A9)
- `/reports`: Final reports
