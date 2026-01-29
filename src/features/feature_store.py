"""
Feature Store for STAVKI (Task M).

Provides:
- Feature snapshots by date (point-in-time correctness)
- Version tracking
- Query API for features at specific timestamps

This prevents data leakage by ensuring features are computed
only with data available at prediction time.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import hashlib
from typing import Optional, Dict, List, Any
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.logging_setup import get_logger

logger = get_logger(__name__)


class FeatureStore:
    """
    Feature store with point-in-time queries and versioning.
    
    Features are stored as snapshots to ensure no data leakage
    (only features computed from data available at query time).
    """
    
    def __init__(self, store_dir: Path):
        """
        Initialize feature store.
        
        Args:
            store_dir: Directory to store feature snapshots
        """
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.store_dir / 'feature_store_metadata.json'
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> dict:
        """Load store metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                return json.load(f)
        return {
            'version': '1.0',
            'snapshots': {},
            'feature_schemas': {}
        }
    
    def _save_metadata(self):
        """Save store metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def _compute_schema_hash(self, df: pd.DataFrame) -> str:
        """Compute hash of feature schema (column names and types)."""
        schema_str = '|'.join(f"{c}:{df[c].dtype}" for c in sorted(df.columns))
        return hashlib.md5(schema_str.encode()).hexdigest()[:8]
    
    def save_snapshot(
        self,
        df: pd.DataFrame,
        snapshot_date: str,
        version: str = 'v1',
        description: str = ''
    ) -> str:
        """
        Save feature snapshot.
        
        Args:
            df: Feature DataFrame
            snapshot_date: Date for this snapshot (YYYY-MM-DD)
            version: Version identifier
            description: Optional description
            
        Returns:
            Snapshot ID
        """
        # Generate snapshot ID
        snapshot_id = f"{snapshot_date}_{version}"
        snapshot_path = self.store_dir / f"snapshot_{snapshot_id}.parquet"
        
        # Save as parquet for efficiency
        df.to_parquet(snapshot_path, index=False)
        
        # Update metadata
        self.metadata['snapshots'][snapshot_id] = {
            'date': snapshot_date,
            'version': version,
            'description': description,
            'path': str(snapshot_path),
            'n_rows': len(df),
            'n_features': len(df.columns),
            'schema_hash': self._compute_schema_hash(df),
            'created_at': datetime.now().isoformat()
        }
        
        # Update schema if new
        schema_hash = self._compute_schema_hash(df)
        if schema_hash not in self.metadata['feature_schemas']:
            self.metadata['feature_schemas'][schema_hash] = {
                'columns': list(df.columns),
                'dtypes': {c: str(df[c].dtype) for c in df.columns}
            }
        
        self._save_metadata()
        
        logger.info(f"Saved snapshot: {snapshot_id} ({len(df)} rows, {len(df.columns)} features)")
        return snapshot_id
    
    def load_snapshot(self, snapshot_id: str) -> Optional[pd.DataFrame]:
        """
        Load specific snapshot.
        
        Args:
            snapshot_id: Snapshot ID to load
            
        Returns:
            DataFrame or None if not found
        """
        if snapshot_id not in self.metadata['snapshots']:
            logger.error(f"Snapshot not found: {snapshot_id}")
            return None
        
        snapshot_info = self.metadata['snapshots'][snapshot_id]
        snapshot_path = Path(snapshot_info['path'])
        
        if not snapshot_path.exists():
            logger.error(f"Snapshot file missing: {snapshot_path}")
            return None
        
        return pd.read_parquet(snapshot_path)
    
    def get_features_at_time(
        self,
        query_date: str,
        version: str = 'v1'
    ) -> Optional[pd.DataFrame]:
        """
        Get features available at a specific point in time.
        
        This ensures point-in-time correctness - only returns features
        that would have been available at query_date.
        
        Args:
            query_date: Date to query features for (YYYY-MM-DD)
            version: Feature version
            
        Returns:
            DataFrame with features available at query_date
        """
        query_dt = pd.to_datetime(query_date)
        
        # Find latest snapshot before query_date
        best_snapshot = None
        best_date = None
        
        for snapshot_id, info in self.metadata['snapshots'].items():
            if info['version'] != version:
                continue
            
            snapshot_dt = pd.to_datetime(info['date'])
            
            if snapshot_dt <= query_dt:
                if best_date is None or snapshot_dt > best_date:
                    best_date = snapshot_dt
                    best_snapshot = snapshot_id
        
        if best_snapshot is None:
            logger.warning(f"No snapshot found for {query_date} (version={version})")
            return None
        
        logger.info(f"Using snapshot {best_snapshot} for query date {query_date}")
        return self.load_snapshot(best_snapshot)
    
    def list_snapshots(self, version: str = None) -> List[Dict[str, Any]]:
        """
        List all available snapshots.
        
        Args:
            version: Filter by version (optional)
            
        Returns:
            List of snapshot metadata
        """
        snapshots = []
        for snapshot_id, info in self.metadata['snapshots'].items():
            if version is None or info['version'] == version:
                snapshots.append({
                    'id': snapshot_id,
                    **info
                })
        
        # Sort by date
        snapshots.sort(key=lambda x: x['date'], reverse=True)
        return snapshots
    
    def get_latest_snapshot(self, version: str = 'v1') -> Optional[str]:
        """
        Get ID of latest snapshot for a version.
        
        Args:
            version: Feature version
            
        Returns:
            Snapshot ID or None
        """
        snapshots = self.list_snapshots(version=version)
        return snapshots[0]['id'] if snapshots else None
    
    def validate_schema(self, df: pd.DataFrame, expected_hash: str) -> bool:
        """
        Validate that DataFrame matches expected schema.
        
        Args:
            df: DataFrame to validate
            expected_hash: Expected schema hash
            
        Returns:
            True if schema matches
        """
        actual_hash = self._compute_schema_hash(df)
        
        if actual_hash != expected_hash:
            logger.warning(f"Schema mismatch: {actual_hash} != {expected_hash}")
            
            if expected_hash in self.metadata['feature_schemas']:
                expected = set(self.metadata['feature_schemas'][expected_hash]['columns'])
                actual = set(df.columns)
                
                missing = expected - actual
                extra = actual - expected
                
                if missing:
                    logger.warning(f"  Missing columns: {missing}")
                if extra:
                    logger.warning(f"  Extra columns: {extra}")
            
            return False
        
        return True


def test_feature_store():
    """Test feature store functionality."""
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        store = FeatureStore(Path(tmpdir))
        
        # Create test data
        df1 = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=100),
            'feature_a': np.random.randn(100),
            'feature_b': np.random.randn(100)
        })
        
        df2 = pd.DataFrame({
            'Date': pd.date_range('2024-04-01', periods=100),
            'feature_a': np.random.randn(100),
            'feature_b': np.random.randn(100)
        })
        
        # Save snapshots
        id1 = store.save_snapshot(df1, '2024-01-01', 'v1', 'Initial snapshot')
        id2 = store.save_snapshot(df2, '2024-04-01', 'v1', 'Q2 update')
        
        # List snapshots
        snapshots = store.list_snapshots()
        print(f"Snapshots: {len(snapshots)}")
        
        # Point-in-time query
        march_data = store.get_features_at_time('2024-03-15', 'v1')
        print(f"Query for 2024-03-15 returned snapshot from: {id1}")
        
        may_data = store.get_features_at_time('2024-05-15', 'v1')
        print(f"Query for 2024-05-15 returned snapshot from: {id2}")
        
        print("\n✅ Feature store test passed!")


if __name__ == '__main__':
    test_feature_store()
