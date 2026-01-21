"""
Data ingestion module for loading and normalizing football match CSV files.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from ..logging_setup import get_logger
from .schemas import (
    FOOTBALL_DATA_COLUMNS,
    ODDS_COLUMNS,
    REQUIRED_COLUMNS,
    NormalizedMatch,
)

logger = get_logger(__name__)


def parse_date(date_str: str) -> Optional[datetime]:
    """
    Parse date string, trying multiple formats.
    
    Args:
        date_str: Date string from CSV
        
    Returns:
        Parsed datetime or None if parsing fails
    """
    # Common date formats in football-data.co.uk
    formats = [
        "%d/%m/%Y",  # 16/08/2025
        "%Y-%m-%d",  # 2025-08-16
        "%d/%m/%y",  # 16/08/25
        "%d-%m-%Y",  # 16-08-2025
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(str(date_str).strip(), fmt)
        except (ValueError, TypeError):
            continue
    
    return None


def load_csv(filepath: Path) -> Optional[pd.DataFrame]:
    """
    Load a single CSV file with error handling.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame or None if loading fails
    """
    try:
        df = pd.read_csv(filepath, encoding="utf-8")
        logger.info(f"Loaded {len(df)} rows from {filepath.name}")
        return df
    except UnicodeDecodeError:
        # Try alternative encoding
        try:
            df = pd.read_csv(filepath, encoding="latin-1")
            logger.info(f"Loaded {len(df)} rows from {filepath.name} (latin-1 encoding)")
            return df
        except Exception as e:
            logger.error(f"Failed to load {filepath.name}: {e}")
            return None
    except Exception as e:
        logger.error(f"Failed to load {filepath.name}: {e}")
        return None


def validate_columns(df: pd.DataFrame, filepath: Path) -> bool:
    """
    Check if DataFrame has required columns.
    
    Args:
        df: DataFrame to validate
        filepath: File path for error messages
        
    Returns:
        True if valid, False otherwise
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    
    if missing:
        logger.warning(
            f"Skipping {filepath.name}: missing required columns {missing}"
        )
        return False
    
    return True


def normalize_matches(df: pd.DataFrame, filepath: Path) -> Tuple[List[NormalizedMatch], int]:
    """
    Normalize DataFrame to NormalizedMatch objects.
    
    Args:
        df: Raw DataFrame from CSV
        filepath: Source file path for logging
        
    Returns:
        Tuple of (list of normalized matches, number of skipped rows)
    """
    matches: List[NormalizedMatch] = []
    skipped = 0
    
    for idx, row in df.iterrows():
        try:
            # Parse date
            parsed_date = parse_date(row["Date"])
            if parsed_date is None:
                logger.warning(
                    f"{filepath.name} row {idx}: Invalid date '{row['Date']}', skipping"
                )
                skipped += 1
                continue
            
            # Parse goals (required)
            try:
                home_goals = int(row["FTHG"])
                away_goals = int(row["FTAG"])
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"{filepath.name} row {idx}: Invalid goals (FTHG={row['FTHG']}, "
                    f"FTAG={row['FTAG']}), skipping"
                )
                skipped += 1
                continue
            
            # Parse odds (optional)
            odds_1 = None
            odds_x = None
            odds_2 = None
            
            if "B365H" in row and pd.notna(row["B365H"]):
                try:
                    odds_1 = float(row["B365H"])
                except (ValueError, TypeError):
                    logger.debug(f"{filepath.name} row {idx}: Invalid B365H, using None")
            
            if "B365D" in row and pd.notna(row["B365D"]):
                try:
                    odds_x = float(row["B365D"])
                except (ValueError, TypeError):
                    logger.debug(f"{filepath.name} row {idx}: Invalid B365D, using None")
            
            if "B365A" in row and pd.notna(row["B365A"]):
                try:
                    odds_2 = float(row["B365A"])
                except (ValueError, TypeError):
                    logger.debug(f"{filepath.name} row {idx}: Invalid B365A, using None")
            
            # Create normalized match
            match = NormalizedMatch(
                date=parsed_date,
                league=str(row["Div"]).strip(),
                home_team=str(row["HomeTeam"]).strip(),
                away_team=str(row["AwayTeam"]).strip(),
                home_goals=home_goals,
                away_goals=away_goals,
                odds_1=odds_1,
                odds_x=odds_x,
                odds_2=odds_2,
            )
            
            matches.append(match)
            
        except Exception as e:
            logger.warning(f"{filepath.name} row {idx}: Error normalizing - {e}, skipping")
            skipped += 1
            continue
    
    logger.info(f"{filepath.name}: Normalized {len(matches)} matches, skipped {skipped} rows")
    return matches, skipped


def deduplicate_matches(matches: List[NormalizedMatch]) -> List[NormalizedMatch]:
    """
    Remove duplicate matches based on (date, home_team, away_team).
    Keeps first occurrence.
    
    Args:
        matches: List of normalized matches
        
    Returns:
        Deduplicated list of matches
    """
    seen_keys = set()
    unique_matches = []
    duplicates = 0
    
    for match in matches:
        key = match.match_key()
        if key not in seen_keys:
            seen_keys.add(key)
            unique_matches.append(match)
        else:
            duplicates += 1
    
    if duplicates > 0:
        logger.info(f"Removed {duplicates} duplicate match(es)")
    
    return unique_matches


def ingest_directory(
    input_dir: Path,
    output_file: Path
) -> dict:
    """
    Ingest all CSV files from directory and save normalized output.
    
    Args:
        input_dir: Directory containing CSV files
        output_file: Path to output CSV file
        
    Returns:
        Dictionary with ingestion statistics
    """
    logger.info(f"Starting ingestion from {input_dir}")
    
    # Ensure input directory exists
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Find all CSV files
    csv_files = list(input_dir.glob("*.csv"))
    
    if not csv_files:
        logger.warning(f"No CSV files found in {input_dir}")
        return {
            "files_processed": 0,
            "total_matches": 0,
            "valid_rows": 0,
            "skipped_rows": 0,
            "duplicates_removed": 0,
        }
    
    logger.info(f"Found {len(csv_files)} CSV file(s)")
    
    # Process all files
    all_matches: List[NormalizedMatch] = []
    files_processed = 0
    total_skipped = 0
    
    for csv_file in sorted(csv_files):
        logger.info(f"Processing file: {csv_file.name}")
        
        # Load CSV
        df = load_csv(csv_file)
        if df is None or df.empty:
            logger.warning(f"Skipping empty or invalid file: {csv_file.name}")
            continue
        
        # Validate columns
        if not validate_columns(df, csv_file):
            continue
        
        # Normalize matches
        matches, skipped = normalize_matches(df, csv_file)
        all_matches.extend(matches)
        total_skipped += skipped
        files_processed += 1
    
    # Deduplicate
    initial_count = len(all_matches)
    all_matches = deduplicate_matches(all_matches)
    duplicates_removed = initial_count - len(all_matches)
    
    # Save to output file
    if all_matches:
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame and save
        output_df = pd.DataFrame([match.to_dict() for match in all_matches])
        output_df.to_csv(output_file, index=False)
        
        logger.info(f"Saved {len(all_matches)} matches to {output_file}")
    else:
        logger.warning("No valid matches to save")
    
    # Return statistics
    stats = {
        "files_processed": files_processed,
        "total_matches": len(all_matches),
        "valid_rows": len(all_matches) + duplicates_removed,
        "skipped_rows": total_skipped,
        "duplicates_removed": duplicates_removed,
    }
    
    logger.info(
        f"Ingestion complete: {files_processed} files, "
        f"{stats['total_matches']} unique matches, "
        f"{stats['skipped_rows']} skipped, "
        f"{duplicates_removed} duplicates removed"
    )
    
    return stats
