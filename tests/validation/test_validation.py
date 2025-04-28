"""Tests for validation methods (WFO, PurgedKFold).""" 

import pytest
import pandas as pd
import numpy as np
from typing import List, Tuple, Iterator, Optional

# Make imports work
import sys
import os
PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from nucleotide_strategy_evolution.validation import (
    generate_wfo_splits,
    PurgedKFold
)

# --- Test Data ---
@pytest.fixture
def time_series_data():
    dates = pd.date_range(start='2023-01-01', periods=100, freq='B') # Business days
    return pd.DataFrame({'Close': np.arange(100)}, index=dates)

# --- Tests for generate_wfo_splits ---

def test_wfo_rolling_basic(time_series_data):
    data = time_series_data
    train_len = 20
    test_len = 5
    step = 5
    splits = list(generate_wfo_splits(data, train_len, test_len, step, anchored=False))
    
    # Calculate expected number of splits
    n = len(data)
    expected_num_splits = (n - train_len - test_len) // step + 1
    assert len(splits) == expected_num_splits
    
    # Check first split
    train0, test0 = splits[0]
    assert len(train0) == train_len
    assert len(test0) == test_len
    assert train0.index.min() == data.index[0]
    assert train0.index.max() == data.index[train_len - 1]
    assert test0.index.min() == data.index[train_len]
    assert test0.index.max() == data.index[train_len + test_len - 1]
    
    # Check second split (shifted by step)
    train1, test1 = splits[1]
    assert len(train1) == train_len
    assert len(test1) == test_len
    assert train1.index.min() == data.index[step]
    assert train1.index.max() == data.index[train_len + step - 1]
    assert test1.index.min() == data.index[train_len + step]
    assert test1.index.max() == data.index[train_len + test_len + step - 1]
    
    # Check last split
    last_train_start_idx = (expected_num_splits - 1) * step
    last_test_end_idx = last_train_start_idx + train_len + test_len
    train_last, test_last = splits[-1]
    assert len(train_last) == train_len
    assert len(test_last) == test_len
    assert train_last.index.min() == data.index[last_train_start_idx]
    assert train_last.index.max() == data.index[last_train_start_idx + train_len - 1]
    assert test_last.index.min() == data.index[last_train_start_idx + train_len]
    assert test_last.index.max() == data.index[last_test_end_idx - 1]
    
def test_wfo_anchored_basic(time_series_data):
    data = time_series_data
    train_len = 20
    test_len = 5
    step = 5
    splits = list(generate_wfo_splits(data, train_len, test_len, step, anchored=True))
    
    n = len(data)
    expected_num_splits = (n - train_len - test_len) // step + 1
    assert len(splits) == expected_num_splits
    
    # Check first split (same as rolling)
    train0, test0 = splits[0]
    assert len(train0) == train_len
    assert len(test0) == test_len
    assert train0.index.min() == data.index[0]
    assert train0.index.max() == data.index[train_len - 1]
    assert test0.index.min() == data.index[train_len]
    assert test0.index.max() == data.index[train_len + test_len - 1]
    
    # Check second split (train starts at 0, ends later)
    train1, test1 = splits[1]
    assert len(train1) == train_len + step # Anchored train grows
    assert len(test1) == test_len
    assert train1.index.min() == data.index[0]
    assert train1.index.max() == data.index[train_len + step - 1]
    assert test1.index.min() == data.index[train_len + step]
    assert test1.index.max() == data.index[train_len + test_len + step - 1]
    
    # Check last split
    last_train_end_idx = (expected_num_splits - 1) * step + train_len
    last_test_end_idx = last_train_end_idx + test_len
    train_last, test_last = splits[-1]
    assert len(train_last) == last_train_end_idx 
    assert len(test_last) == test_len
    assert train_last.index.min() == data.index[0]
    assert train_last.index.max() == data.index[last_train_end_idx - 1]
    assert test_last.index.min() == data.index[last_train_end_idx]
    assert test_last.index.max() == data.index[last_test_end_idx - 1]
    
def test_wfo_step_equals_test(time_series_data):
    # Default behavior if step is None
    data = time_series_data
    train_len = 30
    test_len = 10
    splits_default = list(generate_wfo_splits(data, train_len, test_len, step=None))
    splits_explicit = list(generate_wfo_splits(data, train_len, test_len, step=test_len))
    assert len(splits_default) == len(splits_explicit)
    for i in range(len(splits_default)):
         pd.testing.assert_frame_equal(splits_default[i][0], splits_explicit[i][0])
         pd.testing.assert_frame_equal(splits_default[i][1], splits_explicit[i][1])
         
@pytest.mark.parametrize("train, test, step", [(0, 5, 5), (20, 0, 5), (20, 5, 0), (-1, 5, 5)])
def test_wfo_invalid_periods(time_series_data, train, test, step):
    with pytest.raises(ValueError, match="must be positive"):
        list(generate_wfo_splits(time_series_data, train, test, step))
        
def test_wfo_data_too_short(time_series_data):
    data = time_series_data[:20] # Only 20 data points
    with pytest.raises(ValueError, match="Data length is too short"):
         list(generate_wfo_splits(data, train_periods=15, test_periods=10))
         
def test_wfo_not_datetimeindex():
    data = pd.DataFrame({'Close': range(100)}) # No DatetimeIndex
    with pytest.raises(TypeError, match="Data must have a DatetimeIndex"):
        list(generate_wfo_splits(data, 20, 5))

# --- Tests for PurgedKFold --- 

def test_purged_kfold_init():
    pkf = PurgedKFold(n_splits=5, purge_pct=0.1, embargo_pct=0.05)
    assert pkf.n_splits == 5
    assert pkf.purge_pct == 0.1
    assert pkf.embargo_pct == 0.05
    assert pkf.get_n_splits() == 5

@pytest.mark.parametrize("splits, purge, embargo", [
    (1, 0.1, 0.05),
    (5, -0.1, 0.05),
    (5, 0.6, 0.05),
    (5, 0.1, -0.01),
])
def test_purged_kfold_init_invalid(splits, purge, embargo):
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=splits, purge_pct=purge, embargo_pct=embargo)

def test_purged_kfold_split_basic():
    n_samples = 100
    n_splits = 5
    purge_pct = 0.1
    embargo_pct = 0.05
    data = pd.DataFrame({'value': range(n_samples)}, index=pd.RangeIndex(n_samples))
    
    pkf = PurgedKFold(n_splits=n_splits, purge_pct=purge_pct, embargo_pct=embargo_pct)
    splits = list(pkf.split(data))
    
    assert len(splits) == n_splits
    
    all_train_indices = set()
    all_test_indices = set()
    
    fold_size = n_samples // n_splits # 20
    approx_train_size = n_samples - fold_size # 80
    purge_samples = int(approx_train_size * purge_pct) # 8
    embargo_samples = int(approx_train_size * embargo_pct) # 4
    
    for i, (train_idx, test_idx) in enumerate(splits):
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(test_idx, np.ndarray)
        
        # Check test set boundaries
        expected_test_start = i * fold_size
        expected_test_end = (i + 1) * fold_size if i < n_splits - 1 else n_samples
        assert test_idx[0] == expected_test_start
        assert test_idx[-1] == expected_test_end - 1
        assert len(test_idx) == expected_test_end - expected_test_start
        all_test_indices.update(test_idx)
        
        # Check no overlap between train and test for this fold
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        all_train_indices.update(train_idx)
        
        # Check purging (indices before test set)
        purge_zone_start = max(0, expected_test_start - purge_samples)
        purge_zone_end = expected_test_start
        if purge_zone_end > purge_zone_start:
             purge_zone = np.arange(purge_zone_start, purge_zone_end)
             assert len(np.intersect1d(train_idx, purge_zone)) == 0, f"Fold {i}: Purge failed"
             
        # Check embargo (indices after test set)
        embargo_zone_start = expected_test_end
        embargo_zone_end = min(n_samples, expected_test_end + embargo_samples)
        if embargo_zone_end > embargo_zone_start:
             embargo_zone = np.arange(embargo_zone_start, embargo_zone_end)
             assert len(np.intersect1d(train_idx, embargo_zone)) == 0, f"Fold {i}: Embargo failed"
             
    # Check that all indices were used in testing exactly once
    assert len(all_test_indices) == n_samples
    assert all_test_indices == set(range(n_samples))

def test_purged_kfold_split_no_purge_embargo():
    n_samples = 50
    n_splits = 5
    data = pd.DataFrame({'value': range(n_samples)}, index=pd.RangeIndex(n_samples))
    pkf = PurgedKFold(n_splits=n_splits, purge_pct=0.0, embargo_pct=0.0)
    splits = list(pkf.split(data))
    assert len(splits) == n_splits
    
    # Without purge/embargo, train should contain all indices *not* in test
    for i, (train_idx, test_idx) in enumerate(splits):
        expected_train = np.setdiff1d(np.arange(n_samples), test_idx)
        np.testing.assert_array_equal(np.sort(train_idx), np.sort(expected_train))
        
def test_purged_kfold_split_data_too_small():
    n_samples = 10
    n_splits = 5 # Fold size = 2
    data = pd.DataFrame({'value': range(n_samples)}, index=pd.RangeIndex(n_samples))
    pkf = PurgedKFold(n_splits=n_splits, purge_pct=0.2, embargo_pct=0.1)
    # Approx train size = 8. Purge=1, Embargo=0
    splits = list(pkf.split(data))
    assert len(splits) == n_splits
    # Check first fold: test=[0,1]. Purge before=[]. Embargo after=[2]. Train = [3..9]
    assert np.array_equal(splits[0][0], np.arange(2, 10))
    assert np.array_equal(splits[0][1], np.arange(0, 2))
    # Check last fold: test=[8,9]. Purge before=[7]. Embargo after=[]. Train = [0..6]
    assert np.array_equal(splits[-1][0], np.arange(0, 7))
    assert np.array_equal(splits[-1][1], np.arange(8, 10))
    
def test_purged_kfold_non_monotonic_index():
     data = pd.DataFrame({'value': range(10)}, index=[0,2,1,3,5,4,6,8,7,9])
     pkf = PurgedKFold(n_splits=5)
     with pytest.raises(TypeError, match="must have a monotonically increasing index"):
         list(pkf.split(data)) 