# External References & Research

## Odds API Documentation
- **API Home**: [https://the-odds-api.com/](https://the-odds-api.com/)
- **Markets**: Supports `h2h` (moneyline), `spreads`, `totals`. Default region determines bookmakers (e.g., `us`, `uk`, `eu`).
- **Rate Limits**: 
  - Free/Pro plans have strict quotas.
  - Rate limit headers (`x-requests-remaining`) should be monitored.
  - Status `429` = Too Many Requests.
  - Status `401` = Unauthorized (Check API Key).

## Key Findings for Audit
1.  **H2H Integrity**: The API returns market consensus ("h2h") or specific bookmakers. We implemented `SINGLE_BOOK` mode to ensure 1x2 odds come from a consistent source (Same-Game Parlay compatible integrity) rather than synthetic best-of-market arbitrage which often doesn't exist in reality.
2.  **Timestamps**: The API provides `last_update` per bookmaker. We added `odds_snapshot_time` at ingestion to track exactly when the data was seen, enabling strict time leakage checks.
3.  **Authentication**: Verified key handling via `scripts/diagnostics/odds_api_debug.py`.
