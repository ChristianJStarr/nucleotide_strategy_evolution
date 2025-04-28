"""Functions for strategy validation, including Walk-Forward Optimization."""

import pandas as pd
from typing import List, Tuple, Iterator, Optional
import numpy as np

def generate_wfo_splits(
    data: pd.DataFrame, 
    train_periods: int, 
    test_periods: int, 
    step_periods: Optional[int] = None,
    step: Optional[int] = None,
    anchored: bool = False
) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Generates data splits for Walk-Forward Optimization.

    Args:
        data: The historical data DataFrame (must have a DatetimeIndex).
        train_periods: The number of data points (e.g., days, bars) in each training set.
        test_periods: The number of data points in each testing (OOS) set.
        step_periods: The number of data points to slide the window forward each step.
                      Defaults to `test_periods`.
        step: Alias for step_periods (for backward compatibility).
        anchored: If True, the training window start stays fixed and only the end expands.
                  If False (default), both start and end slide forward.

    Yields:
        Tuples of (train_data, test_data) DataFrames for each WFO step.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("Data must have a DatetimeIndex for WFO.")
        
    # Handle step parameter alias
    if step is not None:
        step_periods = step
        
    if step_periods is None:
        step_periods = test_periods
        
    if train_periods <= 0 or test_periods <= 0 or step_periods <= 0:
        raise ValueError("Train, test, and step periods must be positive.")
        
    n = len(data)
    if n < train_periods + test_periods:
        raise ValueError("Data length is too short for the specified train/test periods.")

    start_idx = 0
    train_end_idx = train_periods
    test_end_idx = train_periods + test_periods

    while test_end_idx <= n:
        if anchored:
             current_train_start_idx = 0 # Start always at the beginning
        else:
             current_train_start_idx = start_idx
             
        train_data = data.iloc[current_train_start_idx:train_end_idx]
        test_data = data.iloc[train_end_idx:test_end_idx]
        
        yield train_data, test_data
        
        # Slide window forward
        start_idx += step_periods
        train_end_idx += step_periods
        test_end_idx += step_periods

# --- Example Usage --- 
if __name__ == '__main__':
    # Create dummy time series data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    dummy_data = pd.DataFrame({
        'Close': range(100)
    }, index=dates)
    
    print("--- Testing Walk-Forward Splits (Rolling Window) ---")
    train_len = 20
    test_len = 5
    step = 5
    split_count = 0
    for train_df, test_df in generate_wfo_splits(dummy_data, train_len, test_len, step, anchored=False):
        split_count += 1
        print(f"Split {split_count}: Train {train_df.index[0].date()} - {train_df.index[-1].date()} ({len(train_df)}), Test {test_df.index[0].date()} - {test_df.index[-1].date()} ({len(test_df)})")
    # Expected splits: 0-19/20-24, 5-24/25-29, ..., 70-89/90-94, 75-94/95-99
    expected_splits = (len(dummy_data) - train_len - test_len) // step + 1
    print(f"Expected number of splits: {expected_splits}")
    assert split_count == expected_splits
    
    print("\n--- Testing Walk-Forward Splits (Anchored Window) ---")
    split_count_anchor = 0
    for train_df, test_df in generate_wfo_splits(dummy_data, train_len, test_len, step, anchored=True):
        split_count_anchor += 1
        print(f"Split {split_count_anchor}: Train {train_df.index[0].date()} - {train_df.index[-1].date()} ({len(train_df)}), Test {test_df.index[0].date()} - {test_df.index[-1].date()} ({len(test_df)})")
    # Expected splits: 0-19/20-24, 0-24/25-29, ... , 0-94/95-99
    print(f"Expected number of splits: {expected_splits}")
    assert split_count_anchor == expected_splits 

# --- Purged K-Fold CV ---

class PurgedKFold:
    """Purged K-Fold Cross-Validator for Time Series Data.

    Splits data into k folds, ensuring that training data does not overlap with
    testing data by removing (purging) data points around the test set boundaries.
    It also supports embargoing, where data immediately following the test set is
    removed from the training set of subsequent folds to prevent leakage from
    autoregressive features.

    Adapted from Marcos Lopez de Prado's work.
    """
    def __init__(self, n_splits: int = 5, purge_pct: float = 0.01, embargo_pct: float = 0.01):
        """
        Args:
            n_splits: Number of folds (k).
            purge_pct: Percentage of the training set size to purge before and after the test set.
            embargo_pct: Percentage of the training set size to embargo (remove from future training) after the test set.
        """
        if not isinstance(n_splits, int) or n_splits < 2:
            raise ValueError("n_splits must be an integer >= 2.")
        if not 0.0 <= purge_pct < 0.5:
            raise ValueError("purge_pct must be between 0.0 and 0.5.")
        if not 0.0 <= embargo_pct:
            raise ValueError("embargo_pct must be non-negative.")
        
        self.n_splits = n_splits
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Returns the number of splitting iterations in the cross-validator."""
        return self.n_splits

    def split(self, X: pd.DataFrame, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generates indices to split data into training and test set.

        Args:
            X: The data DataFrame (must have a monotonically increasing index, preferably DatetimeIndex).
            y: Ignored (for compatibility with scikit-learn API).
            groups: Ignored (for compatibility with scikit-learn API).

        Yields:
            tuple: (train_indices, test_indices) for each split.
        """
        if not isinstance(X.index, (pd.DatetimeIndex, pd.RangeIndex)) or not X.index.is_monotonic_increasing:
            # Relaxing strict DatetimeIndex check, assuming iloc works fine with monotonic index
            if not X.index.is_monotonic_increasing:
                 raise TypeError("X must have a monotonically increasing index.")
                
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Approximate fold size
        fold_size = n_samples // self.n_splits
        if fold_size == 0:
            raise ValueError(f"Cannot split {n_samples} samples into {self.n_splits} folds. Reduce n_splits.")
        
        # Calculate purge and embargo sizes in terms of number of samples
        # Based on the approximate size of the *training* set in a standard K-Fold
        approx_train_size = n_samples - fold_size 
        purge_size = int(approx_train_size * self.purge_pct)
        embargo_size = int(approx_train_size * self.embargo_pct)
        
        current_fold_start = 0
        for k in range(self.n_splits):
            # Determine test set boundaries for this fold
            test_start_idx = current_fold_start
            test_end_idx = test_start_idx + fold_size
            # Adjust the end index for the last fold to include remaining samples
            if k == self.n_splits - 1:
                test_end_idx = n_samples 
                
            # Ensure test_end_idx does not exceed bounds
            test_end_idx = min(test_end_idx, n_samples)
            
            if test_start_idx >= test_end_idx: # Handle cases where fold_size is too large
                 print(f"Warning: Skipping fold {k+1} due to empty test set (start={test_start_idx}, end={test_end_idx}).")
                 continue
                
            test_indices = indices[test_start_idx:test_end_idx]

            # Determine purge boundaries
            purge_start = test_start_idx - purge_size
            purge_end = test_end_idx + purge_size # Purge applied symmetrically in original logic, check Prado
            # Lopez de Prado applies purge only to the *training* set points immediately *before* the test set.
            # Let's adjust: Purge only indices immediately preceding the test set.
            purge_before_start = max(0, test_start_idx - purge_size)
            purge_before_end = test_start_idx
            
            # Determine embargo boundaries (indices to remove from training *after* test set)
            embargo_start = test_end_idx
            embargo_end = min(n_samples, test_end_idx + embargo_size)

            # Define potential training indices (all indices excluding the test set)
            potential_train_indices = np.concatenate((indices[:test_start_idx], indices[test_end_idx:]))
            
            # Apply purging: Remove indices within [purge_before_start, purge_before_end)
            purged_train_indices_mask = \
                (potential_train_indices < purge_before_start) | \
                (potential_train_indices >= purge_before_end) # Keep only outside purge zone before test
                
            # Apply embargo: Remove indices within [embargo_start, embargo_end)
            embargo_mask = \
                (potential_train_indices < embargo_start) | \
                (potential_train_indices >= embargo_end) # Keep only outside embargo zone after test
                
            # Combine masks
            final_train_mask = purged_train_indices_mask & embargo_mask
            train_indices = potential_train_indices[final_train_mask]
            
            if len(train_indices) == 0:
                 print(f"Warning: Skipping fold {k+1} due to empty training set after purging/embargo.")
                 continue
                
            yield train_indices, test_indices

            # Move to the start of the next fold's test set
            current_fold_start = test_end_idx

# --- Example Usage --- 
if __name__ == '__main__':
    # Create dummy time series data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    dummy_data = pd.DataFrame({
        'Close': range(100)
    }, index=dates)
    
    print("--- Testing Walk-Forward Splits (Rolling Window) ---")
    train_len = 20
    test_len = 5
    step = 5
    split_count = 0
    for train_df, test_df in generate_wfo_splits(dummy_data, train_len, test_len, step, anchored=False):
        split_count += 1
        print(f"Split {split_count}: Train {train_df.index[0].date()} - {train_df.index[-1].date()} ({len(train_df)}), Test {test_df.index[0].date()} - {test_df.index[-1].date()} ({len(test_df)})")
    # Expected splits: 0-19/20-24, 5-24/25-29, ..., 70-89/90-94, 75-94/95-99
    expected_splits = (len(dummy_data) - train_len - test_len) // step + 1
    print(f"Expected number of splits: {expected_splits}")
    assert split_count == expected_splits
    
    print("\n--- Testing Walk-Forward Splits (Anchored Window) ---")
    split_count_anchor = 0
    for train_df, test_df in generate_wfo_splits(dummy_data, train_len, test_len, step, anchored=True):
        split_count_anchor += 1
        print(f"Split {split_count_anchor}: Train {train_df.index[0].date()} - {train_df.index[-1].date()} ({len(train_df)}), Test {test_df.index[0].date()} - {test_df.index[-1].date()} ({len(test_df)})")
    # Expected splits: 0-19/20-24, 0-24/25-29, ... , 0-94/95-99
    print(f"Expected number of splits: {expected_splits}")
    assert split_count_anchor == expected_splits

    print("\n--- Testing Purged K-Fold CV ---")
    n_samples_pkf = 100
    k_folds = 5
    purge_p = 0.1 # 10% purge
    embargo_p = 0.05 # 5% embargo
    
    pkf = PurgedKFold(n_splits=k_folds, purge_pct=purge_p, embargo_pct=embargo_p)
    
    # Use the same dummy data, just need the index range
    data_pkf = pd.DataFrame({'value': range(n_samples_pkf)}, index=pd.RangeIndex(n_samples_pkf))
    
    split_num = 0
    max_train_idx_seen_in_test = -1
    min_test_idx_seen_in_train = n_samples_pkf
    
    for fold_idx, (train_idx, test_idx) in enumerate(pkf.split(data_pkf)):
        split_num += 1
        print(f"Fold {fold_idx+1}/{k_folds}: Train size={len(train_idx)}, Test size={len(test_idx)}")
        print(f"  Test indices: {test_idx[0]} - {test_idx[-1]}")
        # print(f"  Train indices: {train_idx}") # Can be long
        
        # Basic checks for purging and embargo
        # Check 1: No test index should be in the train index
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        
        # Check 2: Purging - Train indices should not immediately precede test indices
        approx_train_size = n_samples_pkf - len(test_idx)
        purge_samples = int(approx_train_size * purge_p)
        purge_zone_before = np.arange(max(0, test_idx[0] - purge_samples), test_idx[0])
        if len(purge_zone_before) > 0:
            assert len(np.intersect1d(train_idx, purge_zone_before)) == 0, \
                   f"Fold {fold_idx+1}: Purging failed. Train indices found in [{purge_zone_before[0]}, {purge_zone_before[-1]}]"

        # Check 3: Embargo - Train indices should not immediately follow test indices
        embargo_samples = int(approx_train_size * embargo_p)
        embargo_zone_after = np.arange(test_idx[-1] + 1, min(n_samples_pkf, test_idx[-1] + 1 + embargo_samples))
        if len(embargo_zone_after) > 0:
             assert len(np.intersect1d(train_idx, embargo_zone_after)) == 0, \
                    f"Fold {fold_idx+1}: Embargo failed. Train indices found in [{embargo_zone_after[0]}, {embargo_zone_after[-1]}]"
        
        # Track overlap for general understanding (not a strict test of purge/embargo alone)
        if len(train_idx) > 0:
            max_train_idx_seen_in_test = max(max_train_idx_seen_in_test, train_idx.max() if train_idx.max() < test_idx[0] else -1)
            min_test_idx_seen_in_train = min(min_test_idx_seen_in_train, test_idx.min() if test_idx.min() > train_idx.max() else n_samples_pkf)
            
    print(f"\nTotal splits generated: {split_num}")
    assert split_num == k_folds
    print(f"Max train index before earliest test start (sanity check): {max_train_idx_seen_in_test}")
    print(f"Min test index after latest train end (sanity check): {min_test_idx_seen_in_train}") 