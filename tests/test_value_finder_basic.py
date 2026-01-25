"""
Tests for live value bet finder.
"""

import csv
import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.strategy.value_live import (
    load_latest_odds,
    select_best_prices,
    compute_no_vig_probs,
    compute_ev_candidates,
    rank_value_bets,
    save_value_bets,
)
from src.integration.telegram_notify import format_value_message, is_telegram_configured


@pytest.fixture
def sample_odds_df():
    """Create sample odds DataFrame for testing."""
    data = [
        {
            'event_id': 'evt1',
            'sport_key': 'soccer_epl',
            'commence_time': '2026-01-25T15:00:00Z',
            'home_team': 'Arsenal',
            'away_team': 'Chelsea',
            'bookmaker_key': 'bet365',
            'bookmaker_title': 'Bet365',
            'market_key': 'h2h',
            'outcome_name': 'Arsenal',
            'outcome_price': 2.10,
        },
        {
            'event_id': 'evt1',
            'sport_key': 'soccer_epl',
            'commence_time': '2026-01-25T15:00:00Z',
            'home_team': 'Arsenal',
            'away_team': 'Chelsea',
            'bookmaker_key': 'pinnacle',
            'bookmaker_title': 'Pinnacle',
            'market_key': 'h2h',
            'outcome_name': 'Arsenal',
            'outcome_price': 2.15,  # Better odds
        },
        {
            'event_id': 'evt1',
            'sport_key': 'soccer_epl',
            'commence_time': '2026-01-25T15:00:00Z',
            'home_team': 'Arsenal',
            'away_team': 'Chelsea',
            'bookmaker_key': 'bet365',
            'bookmaker_title': 'Bet365',
            'market_key': 'h2h',
            'outcome_name': 'Chelsea',
            'outcome_price': 3.50,
        },
        {
            'event_id': 'evt1',
            'sport_key': 'soccer_epl',
            'commence_time': '2026-01-25T15:00:00Z',
            'home_team': 'Arsenal',
            'away_team': 'Chelsea',
            'bookmaker_key': 'pinnacle',
            'bookmaker_title': 'Pinnacle',
            'market_key': 'h2h',
            'outcome_name': 'Chelsea',
            'outcome_price': 3.60,  # Better odds
        },
        {
            'event_id': 'evt1',
            'sport_key': 'soccer_epl',
            'commence_time': '2026-01-25T15:00:00Z',
            'home_team': 'Arsenal',
            'away_team': 'Chelsea',
            'bookmaker_key': 'bet365',
            'bookmaker_title': 'Bet365',
            'market_key': 'h2h',
            'outcome_name': 'Draw',
            'outcome_price': 3.40,
        },
    ]
    return pd.DataFrame(data)


@pytest.fixture
def sample_value_bets():
    """Sample value bets for testing formatting."""
    return [
        {
            'event_id': 'evt1',
            'home_team': 'Arsenal',
            'away_team': 'Chelsea',
            'selection': 'Arsenal',
            'odds': 2.15,
            'bookmaker': 'Pinnacle',
            'p_model': 0.52,
            'p_implied': 0.465,
            'ev': 0.118,
            'ev_pct': 11.8,
            'stake': 50.0,
        },
        {
            'event_id': 'evt2',
            'home_team': 'Liverpool',
            'away_team': 'Man City',
            'selection': 'Draw',
            'odds': 3.80,
            'bookmaker': 'Bet365',
            'p_model': 0.30,
            'p_implied': 0.263,
            'ev': 0.14,
            'ev_pct': 14.0,
            'stake': 20.0,
        },
    ]


def test_select_best_prices(sample_odds_df):
    """Test that best prices are selected correctly."""
    best = select_best_prices(sample_odds_df)
    
    # Should have 3 outcomes (Arsenal, Chelsea, Draw) with best odds
    assert len(best) == 3
    
    # Check Arsenal has best odds from Pinnacle
    arsenal_row = best[best['outcome_name'] == 'Arsenal'].iloc[0]
    assert arsenal_row['outcome_price'] == 2.15
    assert arsenal_row['bookmaker_title'] == 'Pinnacle'
    
    # Check Chelsea has best odds from Pinnacle
    chelsea_row = best[best['outcome_name'] == 'Chelsea'].iloc[0]
    assert chelsea_row['outcome_price'] == 3.60
    assert chelsea_row['bookmaker_title'] == 'Pinnacle'


def test_compute_no_vig_probs(sample_odds_df):
    """Test no-vig probability calculation."""
    best = select_best_prices(sample_odds_df)
    no_vig = compute_no_vig_probs(best)
    
    assert 'evt1' in no_vig
    assert 'Arsenal' in no_vig['evt1']
    assert 'Chelsea' in no_vig['evt1']
    assert 'Draw' in no_vig['evt1']
    
    # Sum should be close to 1.0
    total = sum(no_vig['evt1'].values())
    assert abs(total - 1.0) < 0.001


def test_compute_ev_candidates():
    """Test EV candidate computation."""
    # Mock model probs
    model_probs = {
        'evt1': {
            'Arsenal': 0.55,  # Model thinks Arsenal more likely
            'Chelsea': 0.25,
            'Draw': 0.20,
        }
    }
    
    # Mock best prices
    best_prices = pd.DataFrame([
        {
            'event_id': 'evt1',
            'sport_key': 'soccer_epl',
            'commence_time': '2026-01-25T15:00:00Z',
            'home_team': 'Arsenal',
            'away_team': 'Chelsea',
            'bookmaker_key': 'pinnacle',
            'bookmaker_title': 'Pinnacle',
            'market_key': 'h2h',
            'outcome_name': 'Arsenal',
            'outcome_price': 2.20,  # EV = 0.55 * 2.20 - 1 = 0.21
        },
    ])
    
    candidates = compute_ev_candidates(model_probs, best_prices, threshold=0.05)
    
    assert len(candidates) == 1
    assert candidates[0]['selection'] == 'Arsenal'
    assert candidates[0]['ev'] > 0.05


def test_rank_value_bets(sample_value_bets):
    """Test ranking and limiting value bets."""
    # Reverse to test sorting
    reversed_bets = list(reversed(sample_value_bets))
    
    ranked = rank_value_bets(reversed_bets)
    
    # Should be sorted by EV descending
    assert ranked[0]['ev_pct'] > ranked[1]['ev_pct']
    assert ranked[0]['selection'] == 'Draw'  # Higher EV
    
    # Test top_n limit
    top_1 = rank_value_bets(reversed_bets, top_n=1)
    assert len(top_1) == 1
    assert top_1[0]['selection'] == 'Draw'


def test_save_value_bets(sample_value_bets):
    """Test saving value bets to files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path, json_path = save_value_bets(
            sample_value_bets,
            sport='soccer_epl',
            output_dir=tmpdir
        )
        
        # Check files exist
        assert csv_path.exists()
        assert json_path.exists()
        
        # Check CSV content
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2
            assert rows[0]['selection'] == 'Arsenal'
        
        # Check JSON content
        with open(json_path, 'r') as f:
            data = json.load(f)
            assert data['sport'] == 'soccer_epl'
            assert data['count'] == 2
            assert len(data['bets']) == 2


def test_save_value_bets_empty():
    """Test saving empty value bets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path, json_path = save_value_bets(
            [],
            sport='soccer_epl',
            output_dir=tmpdir
        )
        
        # Files should still be created
        assert csv_path.exists()
        assert json_path.exists()
        
        # JSON should indicate zero bets
        with open(json_path, 'r') as f:
            data = json.load(f)
            assert data['count'] == 0


def test_format_value_message(sample_value_bets):
    """Test Telegram message formatting."""
    message = format_value_message(sample_value_bets, top_n=2)
    
    assert 'VALUE BETS FOUND' in message
    assert 'Arsenal' in message
    assert 'Draw' in message
    assert '11.8%' in message or '14.0%' in message
    assert 'Pinnacle' in message or 'Bet365' in message


def test_format_value_message_empty():
    """Test formatting with no value bets."""
    message = format_value_message([])
    assert 'No value bets found' in message


def test_is_telegram_configured():
    """Test Telegram configuration check."""
    # Save original env vars
    original_token = os.environ.get('TELEGRAM_BOT_TOKEN')
    original_chat = os.environ.get('TELEGRAM_CHAT_ID')
    
    try:
        # Test with no config
        os.environ.pop('TELEGRAM_BOT_TOKEN', None)
        os.environ.pop('TELEGRAM_CHAT_ID', None)
        assert is_telegram_configured() is False
        
        # Test with partial config
        os.environ['TELEGRAM_BOT_TOKEN'] = 'test_token'
        assert is_telegram_configured() is False
        
        # Test with full config
        os.environ['TELEGRAM_CHAT_ID'] = 'test_chat'
        assert is_telegram_configured() is True
        
    finally:
        # Restore original env vars
        if original_token:
            os.environ['TELEGRAM_BOT_TOKEN'] = original_token
        else:
            os.environ.pop('TELEGRAM_BOT_TOKEN', None)
        
        if original_chat:
            os.environ['TELEGRAM_CHAT_ID'] = original_chat
        else:
            os.environ.pop('TELEGRAM_CHAT_ID', None)


def test_load_latest_odds_no_files():
    """Test loading odds when no files exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = load_latest_odds('soccer_epl', tmpdir)
        assert result is None


def test_load_latest_odds_with_files():
    """Test loading the latest odds file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        file1 = Path(tmpdir) / 'normalized_soccer_epl_20260120_120000.csv'
        file2 = Path(tmpdir) / 'normalized_soccer_epl_20260122_150000.csv'
        
        # Write minimal CSV
        for f in [file1, file2]:
            with open(f, 'w') as fh:
                fh.write('event_id,outcome_price\n')
                fh.write('evt1,2.0\n')
        
        # Should load the latest file
        df = load_latest_odds('soccer_epl', tmpdir)
        assert df is not None
        assert '_source_file' in df.columns
        assert 'normalized_soccer_epl_20260122_150000.csv' in df['_source_file'].iloc[0]
