"""
Tests for deduplication store.
"""

import os
import tempfile
from pathlib import Path

import pytest

from src.state.dedup_store import DedupStore


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    yield db_path
    
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


def test_dedup_store_creation(temp_db):
    """Test that dedup store creates database and tables."""
    store = DedupStore(temp_db)
    
    assert Path(temp_db).exists()
    
    # Check stats
    stats = store.get_stats()
    assert stats['total_entries'] == 0


def test_record_and_check_duplicate(temp_db):
    """Test recording and checking duplicates."""
    store = DedupStore(temp_db)
    
    # Not a duplicate initially
    assert store.is_duplicate('evt1', 'h2h', 'Arsenal', 'bet365', 2.10) is False
    
    # Record it
    store.record_sent('evt1', 'h2h', 'Arsenal', 'bet365', 2.10, 15.5)
    
    # Now it's a duplicate
    assert store.is_duplicate('evt1', 'h2h', 'Arsenal', 'bet365', 2.10) is True


def test_price_bucketing(temp_db):
    """Test that price bucketing works for minor variations."""
    store = DedupStore(temp_db)
    
    # Record at 2.10
    store.record_sent('evt1', 'h2h', 'Arsenal', 'bet365', 2.10, 15.5)
    
    # 2.10 rounds to 2.1 bucket
    assert store.is_duplicate('evt1', 'h2h', 'Arsenal', 'bet365', 2.10) is True
    
    # 2.15 also rounds to 2.1 or 2.2 (both within threshold, check actual behavior)
    # Price bucketing uses 0.1 increments for odds < 10
    # So 2.05 -> 2.0, 2.10 -> 2.1, 2.15 -> 2.2 are different buckets
    # But 2.12 -> 2.1, 2.08 -> 2.1 would be same
    assert store.is_duplicate('evt1', 'h2h', 'Arsenal', 'bet365', 2.12) is True
    
    # 2.25 rounds to 2.2 or 2.3 (different bucket)
    assert store.is_duplicate('evt1', 'h2h', 'Arsenal', 'bet365', 2.25) is False


def test_different_attributes_not_duplicate(temp_db):
    """Test that different attributes are not considered duplicates."""
    store = DedupStore(temp_db)
    
    store.record_sent('evt1', 'h2h', 'Arsenal', 'bet365', 2.10, 15.5)
    
    # Different event
    assert store.is_duplicate('evt2', 'h2h', 'Arsenal', 'bet365', 2.10) is False
    
    # Different market
    assert store.is_duplicate('evt1', 'totals', 'Arsenal', 'bet365', 2.10) is False
    
    # Different outcome
    assert store.is_duplicate('evt1', 'h2h', 'Chelsea', 'bet365', 2.10) is False
    
    # Different bookmaker
    assert store.is_duplicate('evt1', 'h2h', 'Arsenal', 'pinnacle', 2.10) is False


def test_filter_new_bets(temp_db):
    """Test filtering new bets from a list."""
    store = DedupStore(temp_db)
    
    bets = [
        {
            'event_id': 'evt1',
            'market': 'h2h',
            'selection': 'Arsenal',
            'bookmaker_key': 'bet365',
            'odds': 2.10,
            'ev_pct': 15.5,
        },
        {
            'event_id': 'evt2',
            'market': 'h2h',
            'selection': 'Chelsea',
            'bookmaker_key': 'pinnacle',
            'odds': 3.00,
            'ev_pct': 20.0,
        },
    ]
    
    # All new initially
    new_bets = store.filter_new_bets(bets)
    assert len(new_bets) == 2
    
    # Record one
    store.record_sent('evt1', 'h2h', 'Arsenal', 'bet365', 2.10, 15.5)
    
    # Now one is filtered
    new_bets = store.filter_new_bets(bets)
    assert len(new_bets) == 1
    assert new_bets[0]['event_id'] == 'evt2'


def test_cleanup_old(temp_db):
    """Test cleanup of old entries."""
    import sqlite3
    from datetime import datetime, timedelta
    
    store = DedupStore(temp_db)
    
    # Record a bet
    store.record_sent('evt1', 'h2h', 'Arsenal', 'bet365', 2.10, 15.5)
    
    # Manually backdate it
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    old_time = datetime.now() - timedelta(days=10)
    cursor.execute("""
        UPDATE sent_alerts
        SET sent_at = ?
        WHERE event_id = 'evt1'
    """, (old_time,))
    conn.commit()
    conn.close()
    
    # Cleanup should remove it
    deleted = store.cleanup_old(days=7)
    assert deleted == 1
    
    # Stats should show 0 entries
    stats = store.get_stats()
    assert stats['total_entries'] == 0


def test_stats(temp_db):
    """Test statistics retrieval."""
    store = DedupStore(temp_db)
    
    # Record some bets
    store.record_sent('evt1', 'h2h', 'Arsenal', 'bet365', 2.10, 15.5)
    store.record_sent('evt2', 'h2h', 'Chelsea', 'pinnacle', 3.00, 20.0)
    
    stats = store.get_stats()
    
    assert stats['total_entries'] == 2
    assert stats['last_24h'] == 2
    assert stats['last_48h'] == 2
    assert stats['oldest_entry'] is not None
    assert stats['newest_entry'] is not None
    # temp_db has random name, just check it's a .db file
    assert stats['db_path'].endswith('.db')
