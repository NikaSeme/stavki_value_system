"""
Unit tests for data ingestion module.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from src.data.ingestion import (
    deduplicate_matches,
    ingest_directory,
    load_csv,
    normalize_matches,
    parse_date,
    validate_columns,
)
from src.data.schemas import NormalizedMatch


class TestDateParsing:
    """Test date parsing functionality."""
    
    def test_parse_date_dmy_slash(self):
        """Test DD/MM/YYYY format."""
        result = parse_date("16/08/2025")
        assert result == datetime(2025, 8, 16)
    
    def test_parse_date_ymd_dash(self):
        """Test YYYY-MM-DD format."""
        result = parse_date("2025-08-16")
        assert result == datetime(2025, 8, 16)
    
    def test_parse_date_invalid(self):
        """Test invalid date returns None."""
        result = parse_date("invalid-date")
        assert result is None
    
    def test_parse_date_empty(self):
        """Test empty string returns None."""
        result = parse_date("")
        assert result is None


class TestCSVLoading:
    """Test CSV loading functionality."""
    
    def test_load_csv_valid(self, tmp_path):
        """Test loading valid CSV file."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("Div,Date,HomeTeam,AwayTeam,FTHG,FTAG\nE0,16/08/2025,Arsenal,Chelsea,2,1\n")
        
        df = load_csv(csv_file)
        assert df is not None
        assert len(df) == 1
        assert "Div" in df.columns
    
    def test_load_csv_nonexistent(self, tmp_path):
        """Test loading non-existent file returns None."""
        csv_file = tmp_path / "nonexistent.csv"
        df = load_csv(csv_file)
        assert df is None


class TestColumnValidation:
    """Test column validation."""
    
    def test_validate_columns_valid(self, tmp_path):
        """Test validation with all required columns."""
        df = pd.DataFrame({
            "Div": ["E0"],
            "Date": ["16/08/2025"],
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Chelsea"],
            "FTHG": [2],
            "FTAG": [1]
        })
        assert validate_columns(df, tmp_path / "test.csv") is True
    
    def test_validate_columns_missing(self, tmp_path):
        """Test validation with missing required columns."""
        df = pd.DataFrame({
            "Div": ["E0"],
            "Date": ["16/08/2025"]
        })
        assert validate_columns(df, tmp_path / "test.csv") is False


class TestMatchNormalization:
    """Test match normalization."""
    
    def test_normalize_matches_basic(self, tmp_path):
        """Test basic normalization."""
        df = pd.DataFrame({
            "Div": ["E0"],
            "Date": ["16/08/2025"],
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Chelsea"],
            "FTHG": [2],
            "FTAG": [1],
            "B365H": [1.85],
            "B365D": [3.6],
            "B365A": [4.2]
        })
        
        matches, skipped = normalize_matches(df, tmp_path / "test.csv")
        
        assert len(matches) == 1
        assert skipped == 0
        assert matches[0].league == "E0"
        assert matches[0].home_team == "Arsenal"
        assert matches[0].away_team == "Chelsea"
        assert matches[0].home_goals == 2
        assert matches[0].away_goals == 1
        assert matches[0].odds_1 == 1.85
        assert matches[0].date == datetime(2025, 8, 16)
    
    def test_normalize_matches_missing_odds(self, tmp_path):
        """Test normalization with missing odds."""
        df = pd.DataFrame({
            "Div": ["E0"],
            "Date": ["16/08/2025"],
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Chelsea"],
            "FTHG": [2],
            "FTAG": [1]
        })
        
        matches, skipped = normalize_matches(df, tmp_path / "test.csv")
        
        assert len(matches) == 1
        assert matches[0].odds_1 is None
        assert matches[0].odds_x is None
        assert matches[0].odds_2 is None
    
    def test_normalize_matches_invalid_date(self, tmp_path):
        """Test that invalid dates are skipped."""
        df = pd.DataFrame({
            "Div": ["E0"],
            "Date": ["invalid"],
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Chelsea"],
            "FTHG": [2],
            "FTAG": [1]
        })
        
        matches, skipped = normalize_matches(df, tmp_path / "test.csv")
        
        assert len(matches) == 0
        assert skipped == 1
    
    def test_normalize_matches_invalid_goals(self, tmp_path):
        """Test that invalid goals are skipped."""
        df = pd.DataFrame({
            "Div": ["E0"],
            "Date": ["16/08/2025"],
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Chelsea"],
            "FTHG": ["invalid"],
            "FTAG": [1]
        })
        
        matches, skipped = normalize_matches(df, tmp_path / "test.csv")
        
        assert len(matches) == 0
        assert skipped == 1


class TestDeduplication:
    """Test deduplication functionality."""
    
    def test_deduplicate_matches(self):
        """Test removing duplicate matches."""
        matches = [
            NormalizedMatch(
                date=datetime(2025, 8, 16),
                league="E0",
                home_team="Arsenal",
                away_team="Chelsea",
                home_goals=2,
                away_goals=1,
                odds_1=1.85
            ),
            NormalizedMatch(
                date=datetime(2025, 8, 16),
                league="E0",
                home_team="Arsenal",
                away_team="Chelsea",
                home_goals=2,
                away_goals=1,
                odds_1=1.90  # Different odds, same match
            ),
            NormalizedMatch(
                date=datetime(2025, 8, 17),
                league="E0",
                home_team="Liverpool",
                away_team="Man City",
                home_goals=1,
                away_goals=1
            )
        ]
        
        unique = deduplicate_matches(matches)
        
        assert len(unique) == 2
        assert unique[0].odds_1 == 1.85  # First occurrence kept


class TestFullIngestion:
    """Test complete ingestion workflow."""
    
    def test_ingest_directory(self, tmp_path):
        """Test full directory ingestion."""
        # Create test CSV files
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        
        # File 1
        csv1 = raw_dir / "E0.csv"
        csv1.write_text(
            "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A\n"
            "E0,16/08/2025,Arsenal,Chelsea,2,1,1.85,3.6,4.2\n"
            "E0,17/08/2025,Liverpool,Man City,1,1,2.7,3.4,2.55\n"
        )
        
        # File 2
        csv2 = raw_dir / "E1.csv"
        csv2.write_text(
            "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A\n"
            "E1,18/08/2025,Leeds,Burnley,3,0,2.1,3.1,3.5\n"
        )
        
        output_file = tmp_path / "processed" / "matches.csv"
        
        stats = ingest_directory(raw_dir, output_file)
        
        assert stats["files_processed"] == 2
        assert stats["total_matches"] == 3
        assert stats["skipped_rows"] == 0
        assert output_file.exists()
        
        # Verify output
        output_df = pd.read_csv(output_file)
        assert len(output_df) == 3
        assert list(output_df.columns) == [
            "date", "league", "home_team", "away_team",
            "home_goals", "away_goals", "odds_1", "odds_x", "odds_2"
        ]
