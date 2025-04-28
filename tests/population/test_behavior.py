"""Unit tests for the population behavior characterization module."""

import pytest
import pandas as pd
import numpy as np
from typing import Optional, Dict

# Make sure the package root is in sys.path for imports
import sys
import os
PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from nucleotide_strategy_evolution.population.behavior import (
    characterize_behavior,
    BehaviorVector
)
from nucleotide_strategy_evolution.backtesting.interface import BacktestingResults

# --- Helper to create results --- 
def create_mock_results(trades_data: Optional[Dict] = None) -> BacktestingResults:
    results = BacktestingResults()
    if trades_data:
        results.trades = pd.DataFrame(trades_data)
    else:
        results.trades = pd.DataFrame(columns=['entry_time', 'exit_time', 'pnl'])
    return results

# --- Tests --- 

def test_characterize_behavior_no_trades():
    results = create_mock_results()
    bv = characterize_behavior(results)
    assert bv is None

def test_characterize_behavior_missing_columns():
    results = create_mock_results({'entry_time': [pd.Timestamp('2023-01-01')], 'pnl': [10]})
    bv = characterize_behavior(results)
    assert bv is None

def test_characterize_behavior_basic():
    entry = pd.to_datetime(['2023-01-01 09:00', '2023-01-01 12:00', '2023-01-02 10:00'])
    exit = pd.to_datetime(['2023-01-01 10:00', '2023-01-01 12:30', '2023-01-02 14:00'])
    pnl = [50, -20, 70]
    results = create_mock_results({'entry_time': entry, 'exit_time': exit, 'pnl': pnl})
    
    bv = characterize_behavior(results)
    assert bv is not None
    assert len(bv) == 4 # Matches number of features defined
    
    # Expected values:
    # Holding times (hrs): 1.0, 0.5, 4.0 -> Avg = 5.5 / 3 = 1.833
    # Num trades: 3
    # Win rate: 2/3 = 0.667
    # Profit Factor: (50+70) / 20 = 120 / 20 = 6.0
    assert bv[0] == pytest.approx(1.833333)
    assert bv[1] == 3.0
    assert bv[2] == pytest.approx(0.666667)
    assert bv[3] == pytest.approx(6.0)

def test_characterize_behavior_all_wins():
    entry = pd.to_datetime(['2023-01-01 09:00', '2023-01-01 12:00'])
    exit = pd.to_datetime(['2023-01-01 10:00', '2023-01-01 12:30'])
    pnl = [50, 70]
    results = create_mock_results({'entry_time': entry, 'exit_time': exit, 'pnl': pnl})
    
    bv = characterize_behavior(results)
    assert bv is not None
    # Win rate: 1.0
    # Profit Factor: (50+70)/0 -> inf -> capped at 10.0
    assert bv[2] == 1.0
    assert bv[3] == 10.0 

def test_characterize_behavior_all_losses():
    entry = pd.to_datetime(['2023-01-01 09:00', '2023-01-01 12:00'])
    exit = pd.to_datetime(['2023-01-01 10:00', '2023-01-01 12:30'])
    pnl = [-50, -70]
    results = create_mock_results({'entry_time': entry, 'exit_time': exit, 'pnl': pnl})
    
    bv = characterize_behavior(results)
    assert bv is not None
    # Win rate: 0.0
    # Profit Factor: 0 / (50+70) = 0.0
    assert bv[2] == 0.0
    assert bv[3] == 0.0

def test_characterize_behavior_invalid_dates():
    # Check handling if date conversion fails
    entry = ['invalid-date', '2023-01-01 12:00']
    exit = ['2023-01-01 10:00', '2023-01-01 12:30']
    pnl = [-50, 70]
    results = create_mock_results({'entry_time': entry, 'exit_time': exit, 'pnl': pnl})
    bv = characterize_behavior(results)
    assert bv is not None
    assert bv[0] == 0.0 # Default holding time on error
    assert bv[1] == 2.0 # Num trades
    assert bv[2] == 0.5 # Win rate
    assert bv[3] == pytest.approx(70.0 / 50.0) # Profit factor 