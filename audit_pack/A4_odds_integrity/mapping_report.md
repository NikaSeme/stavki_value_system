# Odds Mapping Report

## 1X2 Market Mapping
Mapping logic applied during normalization:

- **Home**: `h2h` outcome matching home team name (normalized)
- **Away**: `h2h` outcome matching away team name (normalized)
- **Draw**: `h2h` outcome named 'Draw' (case-insensitive)

### Sample Data
```json
[
  {
    "key": "h2h",
    "last_update": "2026-01-25T04:56:10Z",
    "outcomes": [
      {
        "name": "Aston Villa",
        "price": 3.71
      },
      {
        "name": "Newcastle United",
        "price": 2.1
      },
      {
        "name": "Draw",
        "price": 3.69
      }
    ]
  }
]
```
