"""Functions for characterizing the behavior of trading strategies."""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

# Import necessary structure for type hinting
from nucleotide_strategy_evolution.backtesting.interface import BacktestingResults

# Define the structure of the behavioral characterization vector
# Using a list of floats for simplicity, could be a numpy array or named tuple
BehaviorVector = List[float]

def characterize_behavior(results: BacktestingResults) -> Optional[BehaviorVector]:
    """Calculates a behavioral characterization vector from backtest results.
    
    Args:
        results: The BacktestingResults object containing trade data and stats.
        
    Returns:
        A list of floats representing the behavior, or None if characterization fails 
        (e.g., no trades).
    """
    
    trades_df = results.trades
    stats = results.stats
    
    # Check if there's enough data to characterize
    if trades_df is None or trades_df.empty:
        # Return a default vector for non-trading strategies? Or None?
        # Returning None might be cleaner for QD algorithms.
        # Default vector could be all zeros of a fixed length.
        # Let's return None for now.
        # TODO: Decide on handling for strategies with zero trades.
        return None 
        
    # --- Calculate Behavioral Features --- 
    # Ensure required columns exist (these depend on the backtester output)
    required_cols = ['entry_time', 'exit_time', 'pnl'] # Example required columns
    if not all(col in trades_df.columns for col in required_cols):
        print(f"Warning: Missing required columns in trades DataFrame for behavior characterization. Required: {required_cols}")
        return None
        
    # Feature 1: Average Holding Time (in some unit, e.g., hours or relative to data frequency)
    # Assuming entry_time and exit_time are datetime objects or convertible
    try:
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        holding_times = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 3600 # Example: Hours
        avg_holding_time = holding_times.mean()
    except Exception as e:
        print(f"Warning: Could not calculate holding time: {e}")
        avg_holding_time = 0.0 # Default value
        
    # Feature 2: Number of Trades (simple count)
    num_trades = len(trades_df)
    
    # Feature 3: Win Rate
    wins = (trades_df['pnl'] > 0).sum()
    win_rate = wins / num_trades if num_trades > 0 else 0.0
    
    # Feature 4: Profit Factor (Ratio of Gross Profit to Gross Loss)
    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf # Handle zero loss
    # Cap profit factor to avoid extreme values in behavior space?
    profit_factor = min(profit_factor, 10.0) # Example cap at 10
    if profit_factor == np.inf:
         profit_factor = 10.0 # Assign cap if infinite
         
    # --- Assemble Behavior Vector --- 
    # Order matters! Keep it consistent.
    behavior_vector = [
        avg_holding_time,
        float(num_trades),
        win_rate,
        profit_factor
    ]
    
    # Normalize or scale features if necessary? Depends on distance metric used later.
    # For now, return raw features.
    
    # Replace any NaN or Inf values that might have occurred
    behavior_vector = [0.0 if not np.isfinite(v) else v for v in behavior_vector]
    
    return behavior_vector


# --- Example Usage ---
if __name__ == '__main__':
    # Create dummy BacktestingResults
    results_with_trades = BacktestingResults()
    entry_times = pd.to_datetime(['2023-01-01 09:00', '2023-01-01 11:00', '2023-01-02 14:00'])
    exit_times = pd.to_datetime(['2023-01-01 10:30', '2023-01-01 11:45', '2023-01-02 14:15'])
    results_with_trades.trades = pd.DataFrame({
        'entry_time': entry_times,
        'exit_time': exit_times,
        'pnl': [100, -50, 25]
    })
    
    results_no_trades = BacktestingResults()
    results_no_trades.trades = pd.DataFrame(columns=['entry_time', 'exit_time', 'pnl'])
    
    print("--- Testing Behavioral Characterization ---")
    
    bv1 = characterize_behavior(results_with_trades)
    print(f"Characterization (with trades): {bv1}")
    # Expected: [avg_hold_hours, 3.0, 2/3, (100+25)/50=2.5]
    
    bv2 = characterize_behavior(results_no_trades)
    print(f"Characterization (no trades): {bv2}") # Expected: None 