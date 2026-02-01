"""
Strict Feature Contract Enforcer (Task C)

Ensures 100% feature alignment between training and inference.
NO silent zeros, NO auto-fill. Hard fail if features don't match.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Default path to feature contract
DEFAULT_CONTRACT_PATH = Path(__file__).parent.parent.parent / "models" / "feature_columns.json"


class FeatureContractError(Exception):
    """Raised when feature contract is violated."""
    pass


class StrictFeatureContract:
    """
    Enforces strict feature alignment between training and inference.
    
    Key principles:
    1. Single source of truth: feature_columns.json
    2. No silent zeros or auto-fill
    3. Hard fail on mismatch
    """
    
    def __init__(self, contract_path: Optional[Path] = None):
        """
        Initialize with feature contract.
        
        Args:
            contract_path: Path to feature_columns.json
        """
        self.contract_path = contract_path or DEFAULT_CONTRACT_PATH
        self._load_contract()
    
    def _load_contract(self):
        """Load feature contract from JSON."""
        if not self.contract_path.exists():
            raise FileNotFoundError(f"Feature contract not found: {self.contract_path}")
        
        with open(self.contract_path) as f:
            contract = json.load(f)
        
        self.version = contract.get("version", "unknown")
        self.features = contract["features"]
        self.categorical = set(contract.get("categorical_features", []))
        self.numeric = set(contract.get("numeric_features", []))
        
        logger.info(f"Loaded feature contract v{self.version}: {len(self.features)} features")
    
    @property
    def feature_count(self) -> int:
        """Number of features in contract."""
        return len(self.features)
    
    @property
    def feature_set(self) -> Set[str]:
        """Set of feature names for fast lookup."""
        return set(self.features)
    
    def validate_dataframe(
        self,
        df: pd.DataFrame,
        mode: str = "strict"
    ) -> pd.DataFrame:
        """
        Validate and reorder DataFrame to match contract.
        
        Args:
            df: DataFrame with features
            mode: "strict" (hard fail) or "report" (log and continue)
        
        Returns:
            DataFrame with features in canonical order
        
        Raises:
            FeatureContractError: If strict mode and mismatch detected
        """
        actual_features = set(df.columns)
        expected_features = self.feature_set
        
        missing = expected_features - actual_features
        extra = actual_features - expected_features
        common = expected_features & actual_features
        
        # Report findings
        if missing or extra:
            msg = (
                f"Feature contract violation:\n"
                f"  Contract: {len(expected_features)} features\n"
                f"  Provided: {len(actual_features)} features\n"
                f"  Common: {len(common)}\n"
                f"  Missing: {len(missing)} - {sorted(missing)[:10]}\n"
                f"  Extra: {len(extra)} - {sorted(extra)[:10]}"
            )
            
            if mode == "strict":
                raise FeatureContractError(msg)
            else:
                logger.warning(msg)
        
        # Select and reorder columns
        result_cols = [c for c in self.features if c in actual_features]
        
        if len(result_cols) != len(self.features):
            if mode == "strict":
                raise FeatureContractError(
                    f"Cannot produce full feature set: {len(result_cols)}/{len(self.features)}"
                )
        
        return df[result_cols]
    
    def validate_feature_vector(
        self,
        features: Dict[str, any]
    ) -> Dict[str, any]:
        """
        Validate a single feature vector (dict).
        
        Args:
            features: Dict of feature_name -> value
        
        Returns:
            Validated and ordered dict
        
        Raises:
            FeatureContractError: If features don't match contract
        """
        actual = set(features.keys())
        expected = self.feature_set
        
        missing = expected - actual
        extra = actual - expected
        
        if missing:
            raise FeatureContractError(
                f"Missing {len(missing)} required features: {sorted(missing)[:5]}..."
            )
        
        if extra:
            logger.debug(f"Ignoring {len(extra)} extra features: {sorted(extra)[:5]}")
        
        # Return in canonical order
        return {k: features[k] for k in self.features if k in features}
    
    def compare_with_columns(
        self,
        columns: List[str]
    ) -> Dict[str, any]:
        """
        Compare provided columns with contract.
        
        Returns:
            Dict with comparison results
        """
        actual = set(columns)
        expected = self.feature_set
        
        return {
            "contract_count": len(expected),
            "provided_count": len(actual),
            "common": sorted(expected & actual),
            "missing": sorted(expected - actual),
            "extra": sorted(actual - expected),
            "match": expected == actual
        }
    
    def get_fillna_values(self) -> Dict[str, any]:
        """
        Get default fill values for NaN only (not for missing columns!).
        
        This should ONLY be used for NaN within existing columns,
        NOT for adding missing columns.
        """
        fillna = {}
        
        for feat in self.numeric:
            fillna[feat] = 0.0
        
        for feat in self.categorical:
            fillna[feat] = "Unknown"
        
        return fillna


def load_contract() -> StrictFeatureContract:
    """Load the default feature contract."""
    return StrictFeatureContract()


def validate_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate training data against feature contract.
    Raises error if columns don't match.
    """
    contract = load_contract()
    return contract.validate_dataframe(df, mode="strict")


def validate_inference_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate inference data against feature contract.
    Raises error if columns don't match.
    """
    contract = load_contract()
    return contract.validate_dataframe(df, mode="strict")
