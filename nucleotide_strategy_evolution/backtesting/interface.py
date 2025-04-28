"""Interface definition for the backtesting engine."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Tuple, TypeVar, TYPE_CHECKING
import pandas as pd # Assuming pandas DataFrame for data
import random
import numpy as np
# Import time for hour comparison
from datetime import time, datetime 
# Import backtesting library components
from backtesting import Strategy, Backtest
# Import indicator functions (or use TA-Lib/other libraries)
try:
    from backtesting.lib import SMA, EMA, RSI, ATR, cross # Add more as needed
except ImportError:
    # Define mock versions for testing purposes
    def SMA(series, period): return pd.Series(0, index=series.index)
    def EMA(series, period): return pd.Series(0, index=series.index)
    def RSI(series, period): return pd.Series(50, index=series.index)
    def ATR(data, period): return pd.Series(1, index=data.index)
    def cross(series1, series2): return False
# Import Gene type hint
from nucleotide_strategy_evolution.core.structures import Gene 
from nucleotide_strategy_evolution.core.structures import Chromosome

# Define a type variable for forward references
T = TypeVar('T', bound='BacktestingResults')

class BacktestingResults:
    """Stores and encapsulates results from a backtest run."""
    
    def __init__(self) -> None:
        """Initialize the BacktestingResults object with default values."""
        self.equity_curve: Optional[pd.Series] = None
        self.trades: Optional[pd.DataFrame] = None
        self.stats: Dict[str, Any] = {}
        # --- Compliance Specific Fields (Phase 2+) ---
        # Flags indicating if a hard rule was violated during the backtest
        self.hard_rule_violation: bool = False 
        # Detailed violation info (optional)
        self.violations: Dict[str, bool] = { 
            'daily_loss_limit': False,
            'max_trailing_drawdown': False,
            'trading_outside_hours': False, # Add new flag
            # Add other specific rule violation flags as needed
        }
        # Data needed for checking compliance (e.g., daily PnL, peak equity)
        self.compliance_data: Dict[str, Any] = {}

class AbstractBacktestEngine(ABC):
    """Abstract interface for backtesting engines."""

    def __init__(self, data: pd.DataFrame, config: Dict[str, Any]):
        """Initialize the engine with historical data and configuration."""
        self.data = data
        self.config = config
        # TODO: Add data and config validation if needed

    @abstractmethod
    def run(self, chromosome: Chromosome, compliance_rules: Dict[str, Any]) -> BacktestingResults:
        """Run a backtest for a given strategy chromosome and compliance rules.

        Args:
            chromosome: The chromosome representing the strategy to backtest.
            compliance_rules: A dictionary containing the compliance rules to enforce.

        Returns:
            An object containing the results of the backtest.
        """
        pass

# --- Concrete Implementation Placeholder (Example) ---
# You would replace this with an adapter for your chosen backtesting library
# (e.g., backtesting.py, vectorbt, Zipline, custom)

class PlaceholderBacktestEngine(AbstractBacktestEngine):
    """A placeholder implementation that doesn't actually run a backtest."""

    def run(self, chromosome: Chromosome, compliance_rules: Dict[str, Any]) -> BacktestingResults:
        print(f"Placeholder: Pretending to run backtest for chromosome with {len(chromosome.genes)} genes.")
        results = BacktestingResults()
        num_simulated_days = 50
        current_equity = self.config.get('initial_capital', 50000)
        peak_equity = current_equity
        daily_pnl = 0
        simulated_equity_curve = [current_equity]
        daily_loss_limit = compliance_rules.get('daily_loss_limit', float('inf'))
        max_trailing_dd_limit = compliance_rules.get('max_trailing_drawdown', float('inf'))
        scaling_plan = compliance_rules.get('scaling_plan', {})
        scaling_enabled = scaling_plan.get('enabled', False)
        scaling_levels = sorted(scaling_plan.get('levels', []), key=lambda x: x.get('threshold', 0))
        current_contracts = 1
        if scaling_enabled and scaling_levels:
            current_contracts = scaling_levels[0].get('contracts', 1)
        
        # --- Allowed Hours Setup ---
        hours_config = compliance_rules.get('allowed_trading_hours', {})
        try:
            start_time_str = hours_config.get('start', '00:00')
            end_time_str = hours_config.get('end', '23:59')
            allowed_start_time = datetime.strptime(start_time_str, '%H:%M').time()
            allowed_end_time = datetime.strptime(end_time_str, '%H:%M').time()
            # Note: Timezone not used in this simple simulation
        except ValueError:
             print("Warning: Invalid time format in allowed_trading_hours. Using 00:00-23:59.")
             allowed_start_time = time(0, 0)
             allowed_end_time = time(23, 59)
        # --- End Allowed Hours Setup ---
        
        for day in range(num_simulated_days):
            # --- Simulate Time of Day for Trading Activity ---
            # Simulate a random time within a typical trading day (e.g., 8 AM to 4 PM)
            # A real backtester would use the actual timestamp of bars/events.
            simulated_hour = random.randint(8, 16) 
            simulated_minute = random.randint(0, 59)
            simulated_trade_time = time(simulated_hour, simulated_minute)
            # --- End Time Simulation ---
            
            # Check if simulated time is within allowed hours
            is_trading_allowed = allowed_start_time <= simulated_trade_time < allowed_end_time
            # Handle overnight sessions where end < start (simplified check)
            if allowed_end_time < allowed_start_time: 
                is_trading_allowed = simulated_trade_time >= allowed_start_time or simulated_trade_time < allowed_end_time

            if is_trading_allowed:
                # Simulate base PnL per contract, then scale by allowed contracts
                base_daily_pnl_per_contract = random.uniform(-500, 750) 
                daily_pnl = base_daily_pnl_per_contract * current_contracts
            else:
                # Simulate no PnL if outside allowed hours
                daily_pnl = 0.0
                # Optional: Flag that a trade *would* have happened outside hours
                # results.violations['trading_outside_hours'] = True 
                # print(f"  SIM: Day {day+1} - Trading skipped outside allowed hours ({simulated_trade_time})")
            
            # Update equity BEFORE checking scaling for the *next* day's contract size
            current_equity += daily_pnl
            simulated_equity_curve.append(current_equity)
            peak_equity = max(peak_equity, current_equity)
            
            # --- Update Allowed Contracts based on Scaling Plan (for NEXT day) ---
            if scaling_enabled and scaling_levels:
                # Calculate profit relative to initial capital or relevant baseline
                # Simplified: Use peak equity as proxy for profit threshold base
                # More accurate would be profit above initial capital + previous thresholds
                current_profit_level = peak_equity - self.config.get('initial_capital', 50000)
                
                allowed_contracts = scaling_levels[0].get('contracts', 1) # Default to base
                for level in reversed(scaling_levels): # Check from highest tier down
                    threshold = level.get('threshold', 0)
                    contracts = level.get('contracts', 1)
                    if current_profit_level >= threshold:
                        allowed_contracts = contracts
                        break # Found the highest applicable tier
                # Apply the calculated allowed contracts for the next iteration
                if allowed_contracts != current_contracts:
                    print(f"  SIM SCALING: Day {day+1} - Contracts updated to {allowed_contracts} (Profit Level: {current_profit_level:.2f})")
                    current_contracts = allowed_contracts
            # --- End Scaling Plan Update ---
            
            # 1. Check Daily Loss Limit
            # Check against scaled daily PnL
            if daily_pnl < -abs(daily_loss_limit):
                print(f"  SIM VIOLATION: Daily Loss Limit hit on day {day+1} (PnL: {daily_pnl:.2f})")
                results.hard_rule_violation = True
                results.violations['daily_loss_limit'] = True
                break # Stop backtest on hard rule violation
                
            # 2. Check Max Trailing Drawdown
            trailing_drawdown = peak_equity - current_equity
            if trailing_drawdown > abs(max_trailing_dd_limit):
                print(f"  SIM VIOLATION: Max Trailing Drawdown hit on day {day+1} (DD: {trailing_drawdown:.2f})")
                results.hard_rule_violation = True
                results.violations['max_trailing_drawdown'] = True
                break # Stop backtest on hard rule violation
                
            # Reset daily PnL for next day (in reality this is more complex)
            daily_pnl = 0 
            
        # --- End Simulation --- 
        
        # In a real implementation:
        # 1. Decode chromosome into strategy logic understandable by the backtester.
        # 2. Configure the backtester (e.g., commissions, slippage from self.config).
        # 3. Run the backtest simulation on self.data.
        # 4. Compliance rule checks integrated within the loop above.
        # 5. Collect and package results into BacktestingResults.
        
        # Populate some basic stats (even if violated, for context)
        results.stats['simulated_final_equity'] = current_equity
        results.stats['simulated_peak_equity'] = peak_equity
        results.equity_curve = pd.Series(simulated_equity_curve) # Store simulated curve
        # --- Add Dummy Trade Data --- 
        # In a real backtester, this would be generated during the run
        if not results.hard_rule_violation and random.random() > 0.1: # 90% chance of trades if no violation
            num_trades = random.randint(5, 50)
            trade_data = {
                'entry_time': pd.to_datetime(np.random.choice(pd.date_range('2023-01-01', periods=num_simulated_days*24, freq='H'), num_trades)),
                'pnl': np.random.normal(loc=10, scale=100, size=num_trades) # Example PnL distribution
            }
            trades_df = pd.DataFrame(trade_data)
            # Simulate exit time shortly after entry
            trades_df['exit_time'] = trades_df['entry_time'] + pd.to_timedelta(np.random.uniform(0.1, 5.0, num_trades), unit='h')
            results.trades = trades_df.sort_values(by='entry_time').reset_index(drop=True)
        else:
            # No trades or violation occurred
            results.trades = pd.DataFrame(columns=['entry_time', 'exit_time', 'pnl'])
        # --- End Dummy Trade Data ---
        # Add other stats as needed
        
        return results

# --- backtesting.py Integration (Phase 4 / Refinement) ---

# Helper for comparing values based on operator string
def check_condition(val1, operator: str, val2) -> bool:
    if operator == '>': return val1 > val2
    if operator == '<': return val1 < val2
    if operator == '==': return val1 == val2
    # TODO: Implement 'crosses_above', 'crosses_below' using historical data access
    # from backtesting import lib; lib.cross(series1, series2)
    print(f"Warning: Operator '{operator}' not fully implemented yet.")
    return False

# Helper function for MACD calculation (simplified, from backtesting.py examples)
# TODO: Move this helper or use a proper library like TA-Lib
def macd_func(series, fast=12, slow=26, signal=9):
    ema_fast = pd.Series(series).ewm(span=fast, adjust=False).mean()
    ema_slow = pd.Series(series).ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    # Return MACD line, Signal line, Histogram (optional)
    # backtesting.py I() expects a single series or tuple of series
    return macd_line, signal_line # Return tuple for Strategy.I

# Helper function for Bollinger Bands (simplified)
# TODO: Move this helper or use a proper library like TA-Lib
def bbands_func(series, period=20, std_dev=2.0):
    sma = pd.Series(series).rolling(period).mean()
    std = pd.Series(series).rolling(period).std()
    if std is None or std.isnull().all(): # Handle cases with insufficient data for std dev
        # Return NaN bands if std dev cannot be calculated
        nan_series = pd.Series(np.nan, index=series.index)
        return nan_series, sma, nan_series
    upper = sma + std * std_dev
    lower = sma - std * std_dev
    # Return Upper Band, Middle Band (SMA), Lower Band
    return upper, sma, lower

# Mapping from our gene indicator names to backtesting.py / TA-Lib functions
# TODO: Expand this mapping
INDICATOR_MAPPING = {
    "SMA": SMA,
    "EMA": EMA,
    "RSI": RSI,
    "ATR": ATR,
    "MACD": macd_func, # Use our helper for now
    "BBANDS": bbands_func, # Use our helper for now
    # Add others like STOCH, ADX etc. if needed and available/implemented
}
# Mapping for data sources
SOURCE_MAPPING = {
    "close": lambda data: data.Close,
    "open": lambda data: data.Open,
    "high": lambda data: data.High,
    "low": lambda data: data.Low,
    "hl2": lambda data: (data.High + data.Low) / 2,
    "hlc3": lambda data: (data.High + data.Low + data.Close) / 3,
    "ohlc4": lambda data: (data.Open + data.High + data.Low + data.Close) / 4,
    "volume": lambda data: data.Volume if 'Volume' in data else pd.Series(0, index=data.index)
}

class GenericDNAStrategy(Strategy):
    """A generic backtesting.py Strategy that interprets a Chromosome.

    This Strategy dynamically initializes indicators and evaluates entry/exit
    rules based on the genes provided in a `Chromosome` object. It also
    enforces compliance rules defined in `compliance_rules`.

    Attributes:
        chromosome_genes: A list of `Gene` objects defining the strategy.
        compliance_rules: A dictionary of compliance rules to enforce.
    """
    chromosome_genes: Optional[List[Gene]] = None # Changed name for clarity
    compliance_rules: Optional[Dict[str, Any]] = None 

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store initialized indicators from self.I() for lookup
        self._initialized_indicators = {} 
        # Store indicator gene params keyed by the name assigned in self.I()
        self._indicator_specs_by_key: Dict[str, Dict] = {} 
        # Placeholder for order genes (populated in init)
        self.order_management_genes = []
        # Add other gene type lists
        self.time_filter_genes = [] # Example for future use
        self.trade_management_genes = [] # For things like trailing stops

        # State for trailing stops
        self._trailing_stop_activated = False
        self._trailing_stop_price = None
        self._entry_price = None # Track entry price for trailing stop calculation
        self._position_high = None # Track high since entry (for long)
        self._position_low = None # Track low since entry (for short)

        self.atr_value = None # Initialize ATR value holder

        if not self.chromosome_genes:
             print("Warning: No chromosome_genes provided to strategy.")
             return

        # --- Sort Genes by Type --- 
        for gene in self.chromosome_genes:
             # TODO: Potentially use gene.expression_level to filter low-expression genes
              if gene.gene_type == "indicator":
                  self.indicator_genes.append(gene)
              elif gene.gene_type == "entry_rule":
                  self.entry_rule_genes.append(gene)
              elif gene.gene_type == "exit_rule":
                  self.exit_rule_genes.append(gene)
              elif gene.gene_type == "risk_management":
                  self.risk_management_genes.append(gene)
              elif gene.gene_type == "order_management": # Add order management gene type
                  self.order_management_genes.append(gene)
              # TODO: Add other gene types (time_filter, order_management, etc.)
              elif gene.gene_type == "time_filter":
                  self.time_filter_genes.append(gene) # Example
              elif gene.gene_type == "trade_management": # Example: trailing stop gene
                  self.trade_management_genes.append(gene)

        # --- Initialize Indicators defined by Genes --- 
        print(f"Found {len(self.indicator_genes)} indicator genes.")
        for i, gene in enumerate(self.indicator_genes):
             params = gene.parameters
             indicator_name = params.get('indicator_name')
             period = params.get('period') # Needed for most indicators
             source_str = params.get('source', 'close') # Default to close
             
             # Generate a unique key for this indicator instance using its parameters
             param_key_string = self._get_param_key_string(params)
             indicator_key = f"ind_{i}_{indicator_name}({param_key_string})"
             
             if indicator_name in INDICATOR_MAPPING or indicator_name in ["MACD", "BBANDS"]: # Check known custom ones too
                 source_data = SOURCE_MAPPING.get(source_str, lambda data: data.Close)(self.data)
                 
                 try:
                     initialized_indicator = None
                     # --- Handle specific indicator parameters --- 
                     if indicator_name in ["SMA", "EMA", "RSI"]:
                         if period is None: raise ValueError("Period missing")
                         indicator_func = INDICATOR_MAPPING[indicator_name]
                         initialized_indicator = self.I(indicator_func, source_data, period, name=indicator_key)
                         print(f"  Initialized {indicator_key}")
                     elif indicator_name == "MACD":
                         fast_p = params.get('fast_period')
                         slow_p = params.get('slow_period')
                         signal_p = params.get('signal_period')
                         if None in [fast_p, slow_p, signal_p]: raise ValueError("MACD periods missing")
                         # Use custom function helper passed to self.I
                         # Name the indicator using the generated key
                         initialized_indicator = self.I(macd_func, source_data, fast=fast_p, slow=slow_p, signal=signal_p, name=indicator_key)
                         print(f"  Initialized {indicator_key} (MACD Line, Signal Line)")
                     elif indicator_name == "BBANDS":
                         std_dev = params.get('std_dev')
                         if None in [period, std_dev]: raise ValueError("BBANDS period/stddev missing")
                         # Use custom function helper passed to self.I
                         initialized_indicator = self.I(bbands_func, source_data, period=period, std_dev=std_dev, name=indicator_key)
                         print(f"  Initialized {indicator_key} (Upper, Middle, Lower Bands)")
                     else:
                          print(f"Warning: Initialization logic missing for indicator: {indicator_name}")
                          
                     # Store the initialized indicator and its parameters
                     if initialized_indicator is not None:
                         self._initialized_indicators[indicator_key] = initialized_indicator
                         self._indicator_specs_by_key[indicator_key] = params
                         
                 except Exception as e:
                     print(f"Error initializing indicator {indicator_key} from gene {params}: {e}")
             else:
                 print(f"Warning: Indicator '{indicator_name}' not found in INDICATOR_MAPPING.")

        # --- Initialize ATR if needed by Risk Management ---
        self.atr_indicator = None
        atr_needed = False
        atr_period = 14 # Default ATR period
        if self.risk_management_genes:
            # Check first risk gene if it uses ATR
            rm_params = self.risk_management_genes[0].parameters
            sl_mode = rm_params.get('stop_loss_mode')
            tp_mode = rm_params.get('take_profit_mode')
            if sl_mode == 'atr' or tp_mode == 'atr':
                atr_needed = True
                atr_period = rm_params.get('atr_period', 14) # Get period from gene or default
        
        if atr_needed:
             try:
                  # ATR requires High, Low, Close data
                  # Define a key for ATR
                  atr_key = f"ATR(period={atr_period})"
                  # Assuming self.data contains these columns
                  self.atr_indicator = self.I(ATR, self.data, atr_period, name=atr_key)
                  # Store ATR like other indicators for consistency
                  # Make sure to include 'indicator_name' in the spec for lookup
                  atr_spec = {'indicator_name': 'ATR', 'period': atr_period}
                  self._initialized_indicators[atr_key] = self.atr_indicator
                  self._indicator_specs_by_key[atr_key] = atr_spec
                  print(f"  Initialized {atr_key}")
             except Exception as e:
                  print(f"Error initializing ATR indicator: {e}")
                  self.atr_indicator = None # Ensure it's None if init fails

        # --- Initialize Compliance Tracking State --- 
        self.peak_equity = self.equity
        self.daily_pnl = 0.0 # Track PnL within the current day
        self.daily_start_equity = self.equity # Equity at start of day
        self.last_day = None
        self.hard_violation_occurred = False
        self.violation_details = {'daily_loss_limit': False, 'max_trailing_drawdown': False}
        self.contracts_allowed = 1 # Start with base contracts
        self.initial_capital = self.equity # Store initial capital for scaling baseline
        # Load compliance rules for easier access
        self.daily_loss_limit_value = abs(self.compliance_rules.get('daily_loss_limit', float('inf')))
        self.max_trailing_dd_limit_value = abs(self.compliance_rules.get('max_trailing_drawdown', float('inf')))
        hours_conf = self.compliance_rules.get('allowed_trading_hours', {})
        try:
             self.allowed_start_time = datetime.strptime(hours_conf.get('start','00:00'), '%H:%M').time()
             self.allowed_end_time = datetime.strptime(hours_conf.get('end','23:59'), '%H:%M').time()
        except ValueError:
             self.allowed_start_time = time(0,0)
             self.allowed_end_time = time(23,59)
        # Scaling plan rules
        self.scaling_plan = self.compliance_rules.get('scaling_plan', {})
        self.scaling_enabled = self.scaling_plan.get('enabled', False)
        self.scaling_levels = sorted(self.scaling_plan.get('levels', []), key=lambda x: x.get('threshold', 0))
        if self.scaling_enabled and self.scaling_levels:
            self.contracts_allowed = self.scaling_levels[0].get('contracts', 1)
            
        # State for trailing stops
        self._trailing_stop_activated = False
        self._trailing_stop_price = None
        self._entry_price = None # Track entry price for trailing stop calculation
        self._position_high = None # Track high since entry (for long)
        self._position_low = None # Track low since entry (for short)

        # Clear state on re-initialization
        self._reset_trailing_stop_state()

    def _reset_trailing_stop_state(self):
        """Resets the state variables used for trailing stop logic."""
        self._trailing_stop_activated = False
        self._trailing_stop_price = None
        self._entry_price = None
        self._position_high = None
        self._position_low = None

    def next(self):
        """Step function called for each bar of data in the backtest."""
        # --- Pre-computation / State Update --- 
        # Current equity/price is implicitly updated by backtesting.py before next() is called
        # Update peak equity
        self.peak_equity = max(self.peak_equity, self.equity)
        
        # Date/Time checks
        current_dt = self.data.index[-1] 
        current_day = current_dt.date()
        current_time = current_dt.time()

        # --- Daily Reset Logic --- 
        if self.last_day is None:
             self.last_day = current_day
             self.daily_start_equity = self.equity # Set equity at start of first day
             
        if current_day != self.last_day:
             self.daily_pnl = 0 # Reset daily PnL tracker
             self.daily_start_equity = self.equity # Record start equity for the new day
             self.last_day = current_day
             # TODO: Add daily loss check from *previous* day if needed?

        # Calculate realized PnL since start of the day
        # Note: self.equity includes unrealized PnL. Need trade data for realized check.
        # Simplification: Use equity change from day start as proxy for daily PnL check
        self.daily_pnl = self.equity - self.daily_start_equity
        
        # --- Check for Trade Closure (before compliance/logic) ---
        # Check if position was closed externally or by SL/TP hit in previous bar
        if self._entry_price is not None and not self.position:
            print(f"Position closed at {current_dt}. Resetting trailing stop state.")
            self._reset_trailing_stop_state()

        # --- Compliance Checks --- 
        # 1. Daily Loss Limit Check
        if self.daily_pnl < -self.daily_loss_limit_value:
             self.hard_violation_occurred = True
             self.violation_details['daily_loss_limit'] = True
             print(f"VIOLATION: Daily Loss Limit at {current_dt} (Daily PnL: {self.daily_pnl:.2f})" )
             self.position.close() # Close position on violation
             return # Stop further processing for this bar
             
        # 2. Trailing Drawdown Check
        trailing_dd = self.peak_equity - self.equity
        if trailing_dd > self.max_trailing_dd_limit_value:
             self.hard_violation_occurred = True
             self.violation_details['max_trailing_drawdown'] = True
             print(f"VIOLATION: Trailing DD at {current_dt} (Peak: {self.peak_equity:.2f}, Current: {self.equity:.2f}, DD: {trailing_dd:.2f})" )
             self.position.close()
             return
            
        # 3. Check Allowed Hours
        is_trading_allowed = self.allowed_start_time <= current_time < self.allowed_end_time
        if self.allowed_end_time < self.allowed_start_time: # Handle overnight
             is_trading_allowed = current_time >= self.allowed_start_time or current_time < self.allowed_end_time
            
        if not is_trading_allowed:
             # If outside allowed hours, potentially close existing positions?
             # Or just prevent new entries? Let's prevent new entries for now.
             # self.position.close() # Optional: Force close outside hours
             pass # Allow holding positions, but entry logic below will check this
             
        # 4. Update Scaling Plan Allowed Contracts (based on previous day's peak?)
        # Scaling plan logic is complex - typically based on End-of-Day profit/equity
        # Simple approach: Check at each bar based on current peak equity
        if self.scaling_enabled and self.scaling_levels:
             current_profit_level = self.peak_equity - self.initial_capital
             allowed_contracts = self.scaling_levels[0].get('contracts', 1)
             for level in reversed(self.scaling_levels):
                  threshold = level.get('threshold', 0)
                  contracts = level.get('contracts', 1)
                  if current_profit_level >= threshold:
                       allowed_contracts = contracts
                       break 
             self.contracts_allowed = allowed_contracts # Update contracts allowed for trades
             
        # --- Stop if violation occurred --- 
        if self.hard_violation_occurred:
            self.position.close()
            return

        # --- Update ATR value ---
        if self.atr_indicator is not None:
            try:
                self.atr_value = self.atr_indicator[-1]
            except IndexError:
                self.atr_value = None # Not enough data yet

        # --- Risk Management Params --- 
        # Determine SL/TP based on first active risk gene found
        # TODO: Add logic to handle multiple risk/trade management genes, potentially averaging or prioritizing
        stop_loss_price = None
        take_profit_price = None
        if self.risk_management_genes:
            rm_gene = self.risk_management_genes[0] # Use the first one found
            # TODO: Consider gene.expression_level
            rm_params = rm_gene.parameters
            sl_mode = rm_params.get('stop_loss_mode')
            sl_value = rm_params.get('stop_loss_value')
            tp_mode = rm_params.get('take_profit_mode')
            tp_value = rm_params.get('take_profit_value')
            
            # Calculate SL/TP Prices (Needs current price & potentially ATR)
            current_price = self.data.Close[-1]
            pip = getattr(self.data._broker, '_pip', 0.0001) # Get pip size if available
            atr_val = self.atr_value # Use the pre-calculated ATR

            # --- Stop Loss Calculation ---
            sl_set = False
            if sl_mode == 'ticks' and sl_value is not None:
                sl_distance = sl_value * pip
                sl_set = True
            elif sl_mode == 'percentage' and sl_value is not None:
                sl_distance = current_price * (sl_value / 100.0)
                sl_set = True
            elif sl_mode == 'atr' and sl_value is not None and atr_val is not None and atr_val > 0:
                sl_distance = sl_value * atr_val
                sl_set = True
            
            if sl_set:
                if self.position.is_long:
                    stop_loss_price = current_price - sl_distance
                elif self.position.is_short:
                    stop_loss_price = current_price + sl_distance
                else: # If position not open yet, calculate based on potential long entry
                    stop_loss_price = current_price - sl_distance 
            # else: SL mode is None or invalid -> sl_price remains None

            # --- Take Profit Calculation ---
            tp_set = False
            if tp_mode == 'ticks' and tp_value is not None:
                tp_distance = tp_value * pip
                tp_set = True
            elif tp_mode == 'percentage' and tp_value is not None: # TP Percentage (Added)
                 tp_distance = current_price * (tp_value / 100.0)
                 tp_set = True
            elif tp_mode == 'atr' and tp_value is not None and atr_val is not None and atr_val > 0:
                 tp_distance = tp_value * atr_val
                 tp_set = True
            elif tp_mode == 'rr_ratio' and tp_value is not None and sl_distance > 0:
                tp_distance = sl_distance * tp_value
                tp_set = True
                
            if tp_set:
                if self.position.is_long:
                    take_profit_price = current_price + tp_distance
                elif self.position.is_short:
                    take_profit_price = current_price - tp_distance
                else: # If position not open yet, calculate based on potential long entry
                     take_profit_price = current_price + tp_distance
            # else: TP mode is None or invalid -> tp_price remains None

        # --- Trailing Stop Parameters ---
        # --- Trailing Stop Parameters (Example using first trade_management gene) ---
        self.trailing_stop_mode = None
        self.trailing_stop_value = None
        self.trailing_stop_atr_period = None # Needed if mode is ATR
        if self.trade_management_genes:
            tm_gene = self.trade_management_genes[0]
            tm_params = tm_gene.parameters
            if tm_params.get('type') == 'trailing_stop':
                self.trailing_stop_mode = tm_params.get('mode') # e.g., 'atr', 'percentage', 'ticks'
                self.trailing_stop_value = tm_params.get('value')
                self.trailing_stop_atr_period = tm_params.get('atr_period') # Optional, if mode is atr
                # Note: ATR for trailing stop might use a different period than SL/TP ATR
                # If so, initialize a separate ATR indicator in init()

        # --- Order Management Params --- 
        # Get params from first active order management gene
        order_type = "Market" 
        limit_p = None # Use different names to avoid conflict with SL/TP
        stop_p = None 
        entry_offset_ticks = 0
        # TODO: Implement TIF handling if needed by backtester
        # time_in_force = "GTC"
        order_genes = [g for g in self.chromosome_genes if g.gene_type == 'order_management']
        if order_genes:
            om_params = order_genes[0].parameters # Use first one found
            order_type = om_params.get('order_type', 'Market')
            entry_offset_ticks = om_params.get('entry_offset_ticks', 0)
            
            # Calculate Limit/Stop Entry Prices based on offset and potential direction
            # This calculation happens *before* knowing the trade direction
            # The actual buy/sell call will use the correct price based on direction
            current_price = self.data.Close[-1]
            pip = getattr(self.data._broker, '_pip', 0.0001) 
            offset_value = entry_offset_ticks * pip

            if order_type == 'Limit':
                # Buy Limit is below current, Sell Limit is above current
                limit_p_buy = current_price - offset_value 
                limit_p_sell = current_price + offset_value
            elif order_type == 'Stop': 
                # Buy Stop is above current, Sell Stop is below current
                 stop_p_buy = current_price + offset_value
                 stop_p_sell = current_price - offset_value

        # --- Strategy Logic --- 
        # --- Evaluate Entry Rules --- 
        should_enter = False
        entry_direction = 'long' # Default direction
        
        if not self.position and is_trading_allowed: # Check if allowed to trade now
            for entry_gene in self.entry_rule_genes:
                # TODO: Consider gene.expression_level
                rule_met = False
                gene_direction = entry_gene.parameters.get('direction', 'long').lower() # Get direction from gene
                conditions = entry_gene.parameters.get('conditions', [])
                logic = entry_gene.parameters.get('logic', 'AND').upper()
                condition_results = []

                for i, cond in enumerate(conditions):
                    cond_met = False
                    component_index = cond.get('component_index') # Index for multi-output indicators (e.g., MACD line vs signal)
                    threshold_component_index = cond.get('threshold_component_index') # Index for threshold indicator
                    try:
                        indicator_name = cond.get('indicator_name')
                        # Use ALL params from the condition dict itself for lookup,
                        # _get_indicator_series will filter irrelevant ones
                        series1 = self._get_indicator_series(cond) 
                        if series1 is None: 
                            #print(f"Debug: series1 None for cond {cond}")
                            continue
                        
                        operator = cond.get('operator')
                        threshold_type = cond.get('threshold_type')
                        
                        if operator in ['crosses_above', 'crosses_below']:
                            val2_series = None # The series to compare against
                            if threshold_type == 'constant':
                                 threshold_value = cond.get('threshold_value')
                                 if threshold_value is not None:
                                     # Select the correct component if series1 is a tuple
                                     s1_comp = series1[component_index or 0] if isinstance(series1, tuple) else series1
                                     cond_met = self._check_crossing(s1_comp, operator, threshold_value)
                            elif threshold_type == 'indicator':
                                 th_name = cond.get('threshold_indicator_name')
                                 # Use threshold params from condition for lookup
                                 # Prefix keys with 'threshold_'
                                 th_cond_params = {k.replace('threshold_', ''): v 
                                                   for k, v in cond.items() 
                                                   if k.startswith('threshold_') and k != 'threshold_type'}
                                 series2 = self._get_indicator_series(th_cond_params)
                                 if series2 is not None:
                                     # Select components based on indices specified in condition
                                     s1_comp = series1[component_index or 0] if isinstance(series1, tuple) else series1
                                     s2_comp = series2[threshold_component_index or 0] if isinstance(series2, tuple) else series2
                                     cond_met = self._check_crossing(s1_comp, operator, s2_comp)
                        else:
                            # Standard comparison
                            val1 = self._get_indicator_value(series1, component_index=component_index) 
                            if val1 is None: 
                                #print(f"Debug: val1 None for cond {cond}")
                                continue 
                                
                            val2 = None
                            if threshold_type == 'constant':
                                val2 = cond.get('threshold_value')
                            elif threshold_type == 'indicator':
                                 th_cond_params = {k.replace('threshold_', ''): v 
                                                   for k, v in cond.items() 
                                                   if k.startswith('threshold_') and k != 'threshold_type'}
                                 series2 = self._get_indicator_series(th_cond_params)
                                 val2 = self._get_indicator_value(series2, component_index=threshold_component_index)
                                 
                            if val1 is not None and val2 is not None:
                                 cond_met = self._check_value_condition(val1, operator, val2)
                                 
                    except Exception as e:
                        print(f"Error evaluating condition {i} in {entry_gene.gene_type}: {e}")
                        
                    condition_results.append(cond_met)

                # Combine results based on logic
                if condition_results:
                    if logic == 'AND': rule_met = all(condition_results)
                    elif logic == 'OR': rule_met = any(condition_results)
                        
                if rule_met:
                    should_enter = True
                    entry_direction = gene_direction # Set direction based on the triggering gene
                    break 
                    
            # Place Order (Buy or Sell)
            if should_enter:
                 entry_kwargs = {
                     'size': self.contracts_allowed,
                     'sl': stop_loss_price,
                     'tp': take_profit_price
                 }
                 
                 # Adjust SL/TP calculation based on intended direction BEFORE placing order
                 current_price = self.data.Close[-1]
                 sl_distance = None
                 tp_distance = None
                 # Recalculate distances based on rm_gene if available
                 if self.risk_management_genes:
                    rm_params = self.risk_management_genes[0].parameters
                    sl_mode = rm_params.get('stop_loss_mode')
                    sl_value = rm_params.get('stop_loss_value')
                    tp_mode = rm_params.get('take_profit_mode')
                    tp_value = rm_params.get('take_profit_value')
                    atr_val = self.atr_value
                    pip = getattr(self.data._broker, '_pip', 0.0001)
                    
                    # Recalculate SL distance
                    if sl_mode == 'ticks' and sl_value is not None: sl_distance = sl_value * pip
                    elif sl_mode == 'percentage' and sl_value is not None: sl_distance = current_price * (sl_value / 100.0)
                    elif sl_mode == 'atr' and sl_value is not None and atr_val is not None and atr_val > 0: sl_distance = sl_value * atr_val
                    
                    # Recalculate TP distance
                    tp_set = False
                    if tp_mode == 'ticks' and tp_value is not None: tp_distance = tp_value * pip; tp_set = True
                    elif tp_mode == 'percentage' and tp_value is not None: tp_distance = current_price * (tp_value / 100.0); tp_set = True
                    elif tp_mode == 'atr' and tp_value is not None and atr_val is not None and atr_val > 0: tp_distance = tp_value * atr_val; tp_set = True
                    elif tp_mode == 'rr_ratio' and tp_value is not None and sl_distance is not None and sl_distance > 0: tp_distance = sl_distance * tp_value; tp_set = True
                    if not tp_set: tp_distance = None # Ensure TP distance is None if not set

                 if entry_direction == 'long':
                      if sl_distance is not None: entry_kwargs['sl'] = current_price - sl_distance
                      if tp_distance is not None: entry_kwargs['tp'] = current_price + tp_distance
                      if order_type == 'Limit': entry_kwargs['limit'] = limit_p_buy
                      elif order_type == 'Stop': entry_kwargs['stop'] = stop_p_buy
                      
                      print(f"Entry condition met at {self.data.index[-1]}. Placing {order_type} LONG order. Args: {entry_kwargs}")
                      self.buy(**entry_kwargs)

                 elif entry_direction == 'short':
                      if sl_distance is not None: entry_kwargs['sl'] = current_price + sl_distance
                      if tp_distance is not None: entry_kwargs['tp'] = current_price - tp_distance
                      if order_type == 'Limit': entry_kwargs['limit'] = limit_p_sell 
                      elif order_type == 'Stop': entry_kwargs['stop'] = stop_p_sell
                      
                      print(f"Entry condition met at {self.data.index[-1]}. Placing {order_type} SHORT order. Args: {entry_kwargs}")
                      self.sell(**entry_kwargs) # Use sell for short entry

        # --- Evaluate Exit Rules --- 
        elif self.position: 
            should_exit = False
            # --- 1. Check Trailing Stop ---
            # TODO: Refine trailing stop logic (e.g., activation threshold, using a separate ATR indicator if period differs)
            if self.trailing_stop_mode and self.trailing_stop_value:
                current_price = self.data.Close[-1]
                pip = getattr(self.data._broker, '_pip', 0.0001)
                atr_val = self.atr_value # Use main ATR for now
                # TODO: If trailing_stop_atr_period is different, fetch the correct ATR indicator value here
                # Calculate trailing stop distance
                ts_distance = None
                if self.trailing_stop_mode == 'ticks': ts_distance = self.trailing_stop_value * pip
                elif self.trailing_stop_mode == 'percentage': 
                     # Percentage of entry price or current price? Let's use entry price for consistency
                     ts_distance = self._entry_price * (self.trailing_stop_value / 100.0) 
                elif self.trailing_stop_mode == 'atr' and atr_val is not None and atr_val > 0:
                     ts_distance = self.trailing_stop_value * atr_val

                if ts_distance is not None:
                     new_ts_price = None
                     if self.position.is_long:
                         new_ts_price = self._position_high - ts_distance
                         # Activate only if TS is above entry price? (Optional)
                         # Refined: Activate only if new TS price offers improvement AND is above entry
                         # Add a small buffer (e.g., 1 tick) to avoid immediate stop-out on entry
                         activation_threshold = self._entry_price + pip # Simple activation threshold
                         if new_ts_price > activation_threshold and \
                            (self._trailing_stop_price is None or new_ts_price > self._trailing_stop_price):
                              self._trailing_stop_price = new_ts_price
                         # Check if activated and price hits the trailing stop
                         if self._trailing_stop_price is not None and current_price <= self._trailing_stop_price:
                              print(f"Trailing Stop LONG hit at {current_price} (TS: {self._trailing_stop_price})")
                              should_exit = True
                     elif self.position.is_short:
                         new_ts_price = self._position_low + ts_distance
                         # Activate only if TS is below entry price? (Optional)
                         activation_threshold = self._entry_price - pip # Simple activation threshold
                         if new_ts_price < activation_threshold and \
                            (self._trailing_stop_price is None or new_ts_price < self._trailing_stop_price):
                              self._trailing_stop_price = new_ts_price
                         # Check if activated and price hits the trailing stop
                         if self._trailing_stop_price is not None and current_price >= self._trailing_stop_price:
                              print(f"Trailing Stop SHORT hit at {current_price} (TS: {self._trailing_stop_price})")
                              should_exit = True
            
            # --- 2. Evaluate Gene-based Exit Rules --- 
            # Only check if not already exiting due to trailing stop
            if not should_exit:
                 for exit_gene in self.exit_rule_genes:
                      rule_met = False 
                      conditions = exit_gene.parameters.get('conditions', [])
                      logic = exit_gene.parameters.get('logic', 'AND').upper()
                      condition_results = []
                      for i, cond in enumerate(conditions):
                           cond_met = False
                           component_index = cond.get('component_index')
                           threshold_component_index = cond.get('threshold_component_index')
                           try:
                               # Use the same robust logic as entry rules
                               indicator_name = cond.get('indicator_name')
                               series1 = self._get_indicator_series(cond)
                               if series1 is None: continue
                               
                               operator = cond.get('operator')
                               threshold_type = cond.get('threshold_type')

                               if operator in ['crosses_above', 'crosses_below']:
                                    if threshold_type == 'constant':
                                         threshold_value = cond.get('threshold_value')
                                         if threshold_value is not None:
                                             s1_comp = series1[component_index or 0] if isinstance(series1, tuple) else series1
                                             cond_met = self._check_crossing(s1_comp, operator, threshold_value)
                                    elif threshold_type == 'indicator':
                                         th_cond_params = {k.replace('threshold_', ''): v 
                                                           for k, v in cond.items() 
                                                           if k.startswith('threshold_') and k != 'threshold_type'}
                                         series2 = self._get_indicator_series(th_cond_params)
                                         if series2 is not None:
                                             s1_comp = series1[component_index or 0] if isinstance(series1, tuple) else series1
                                             s2_comp = series2[threshold_component_index or 0] if isinstance(series2, tuple) else series2
                                             cond_met = self._check_crossing(s1_comp, operator, s2_comp)
                               else:
                                    # Standard comparison
                                    val1 = self._get_indicator_value(series1, component_index=component_index)
                                    if val1 is None: continue
                                    
                                    val2 = None
                                    if threshold_type == 'constant':
                                         val2 = cond.get('threshold_value')
                                    elif threshold_type == 'indicator':
                                         th_cond_params = {k.replace('threshold_', ''): v 
                                                          for k, v in cond.items() 
                                                          if k.startswith('threshold_') and k != 'threshold_type'}
                                         series2 = self._get_indicator_series(th_cond_params)
                                         val2 = self._get_indicator_value(series2, component_index=threshold_component_index)
                                    
                                    if val1 is not None and val2 is not None:
                                         cond_met = self._check_value_condition(val1, operator, val2)
                                 
                           except Exception as e:
                               print(f"Error evaluating exit condition {i}: {e}")
                               
                           condition_results.append(cond_met)
                           
                      if condition_results:
                          if logic == 'AND': rule_met = all(condition_results)
                          elif logic == 'OR': rule_met = any(condition_results)
                          
                      if rule_met:
                          should_exit = True
                          break # Exit based on first met rule

            # --- Close Position if Exit Condition Met ---
            if should_exit:
                 print(f"Exit condition met at {self.data.index[-1]}. Closing position.")
                 self.position.close()
                 self._reset_trailing_stop_state() # Reset state after closing

    # --------------------------------------------------------------------------
    # Helper Methods for Indicator and Condition Handling
    # --------------------------------------------------------------------------

    def _get_param_key_string(self, params: Dict) -> str:
        """Generates a consistent key string from indicator parameters.

        Excludes certain keys and sorts the rest to ensure consistency
        regardless of parameter order in the gene definition.

        Args:
            params: Dictionary of parameters for an indicator.

        Returns:
            A sorted, comma-separated string representation of relevant parameters.
        """
        # Keys to exclude when creating the unique identifier string
        exclude_keys = {
            'indicator_name', 'operator', 'threshold_type', 'threshold_value',
            'threshold_indicator_name', 'threshold_indicator_params',
            'component_index', 'threshold_component_index'
        }
        param_items = sorted([f"{k}={v}" for k, v in params.items() if k not in exclude_keys])
        return ','.join(param_items)

    def _get_indicator_series(self, cond_params: Dict) -> Optional[Any]:
        """Retrieves the initialized indicator series based on condition parameters.

        Finds the indicator initialized in `init` that matches the
        `indicator_name` and other defining parameters specified in the
        `cond_params` dictionary.

        Args:
            cond_params: The parameters dictionary from a rule condition,
                         which specifies the desired indicator and its params.

        Returns:
            The corresponding indicator series object (e.g., a pandas Series or
            a tuple of Series for multi-output indicators) if found,
            otherwise None.
        """
        target_indicator_name = cond_params.get('indicator_name')
        if not target_indicator_name:
            print("Warning: Condition parameters missing 'indicator_name'.")
            return None

        # Generate the parameter key string based *only* on the parameters relevant
        # for identifying this specific indicator instance (e.g., period, source).
        target_param_key_string = self._get_param_key_string(cond_params)

        # Search through the dictionary of indicators initialized in self.init()
        for key, series in self._initialized_indicators.items():
            # Retrieve the original parameters used during initialization
            init_params = self._indicator_specs_by_key.get(key)
            if not init_params:
                print(f"Warning: Missing initialization parameters for indicator key '{key}'.")
                continue # Should not happen if populated correctly

            init_indicator_name = init_params.get('indicator_name')
            init_param_key_string = self._get_param_key_string(init_params)

            # Check if the indicator name and its identifying parameters match
            if (init_indicator_name == target_indicator_name and
                init_param_key_string == target_param_key_string):
                return series # Found the matching initialized indicator series

        # If no match was found after checking all initialized indicators
        search_params = {k: v for k,v in cond_params.items() if k not in ['operator', 'threshold_type', 'threshold_value', 'threshold_indicator_name', 'threshold_indicator_params', 'component_index', 'threshold_component_index']}
        print(f"Warning: Indicator series not found for name '{target_indicator_name}' with params {search_params}")
        print(f"         Available indicators: {list(self._indicator_specs_by_key.keys())}")
        return None

    def _get_indicator_value(self,
                             indicator_series: Any,
                             component_index: Optional[int] = 0) -> Optional[float]:
        """Extracts the latest numerical value from an indicator series.

        Handles both single-output (e.g., Series) and multi-output
        (e.g., tuple of Series like MACD, BBands) indicators.

        Args:
            indicator_series: The indicator object (Series, tuple of Series, etc.).
            component_index: For multi-output indicators, the index of the
                             component series to retrieve the value from (default is 0).

        Returns:
            The latest numerical value as a float, or None if the value cannot
            be retrieved (e.g., not enough data, invalid index).
        """
        if indicator_series is None:
            return None

        try:
            target_series = None
            # If indicator returns a tuple (like MACD, BBands), access the specified component
            if isinstance(indicator_series, tuple):
                idx = component_index if component_index is not None else 0
                if 0 <= idx < len(indicator_series):
                    target_series = indicator_series[idx]
                else:
                    print(f"Warning: Component index {idx} out of bounds for indicator tuple (len {len(indicator_series)}). Required by a condition.")
                    return None
            else:
                # Assume it's a single series
                target_series = indicator_series

            # Check if the target series is array-like and has data
            if hasattr(target_series, '__len__') and len(target_series) > 0:
                # Return the latest value
                value = target_series[-1]
                # Ensure it's a numerical type (float or int) before returning
                if isinstance(value, (int, float, np.number)):
                   # Avoid returning numpy bools if a condition accidentally gets one
                   if isinstance(value, np.bool_):
                        print(f"Warning: Indicator value retrieval got boolean ({value}), expected number. Returning None.")
                        return None
                   return float(value)
                else:
                    # Could be NaN or some other non-numeric type
                    # print(f"Warning: Latest indicator value is not numeric ({type(value)}). Returning None.")
                    return None # Return None for NaN or non-numeric types
            else:
                # Series exists but is empty (e.g., start of backtest)
                # print(f"Warning: Indicator series component is not array-like or is empty.")
                return None

        except IndexError:
             # This might happen if the series is too short (e.g., first few bars)
             # print(f"Warning: IndexError accessing indicator value (likely insufficient data). Returning None.")
             return None # Indicator might not have enough data yet
        except Exception as e:
            print(f"Error getting indicator value: {e}. Returning None.")
            return None

    def _check_value_condition(self,
                               val1: Optional[float],
                               operator: str,
                               val2: Optional[float]) -> bool:
        """Compares two numerical values based on a string operator.

        Handles >, <, == operators for float values. Returns False if either
        value is None.

        Args:
            val1: The first numerical value.
            operator: The comparison operator ('>', '<', '==').
            val2: The second numerical value.

        Returns:
            True if the condition is met, False otherwise.
        """
        if val1 is None or val2 is None:
            # print(f"Debug: Cannot compare values, one is None. val1={val1}, op='{operator}', val2={val2}")
            return False # Cannot compare if values are missing

        try:
            if operator == '>': return val1 > val2
            if operator == '<': return val1 < val2
            if operator == '==': return val1 == val2
            # Add other simple comparisons if needed (e.g., '>=', '<=')
            # Note: Crossing checks require the _check_crossing method using series data.
            print(f"Warning: Operator '{operator}' not supported for direct value comparison. Use '>', '<', or '=='.")
            return False
        except TypeError:
            print(f"Warning: TypeError during value comparison: {val1} {operator} {val2}. Returning False.")
            return False

    def _check_crossing(self,
                        series1: Any,
                        operator: str,
                        series2_or_const: Any) -> bool:
        """Checks for crossing conditions between two series or a series and a constant.

        Uses `backtesting.lib.cross`. Handles 'crosses_above' and
        'crosses_below'. Ensures series are numpy arrays and handles
        comparison against a constant value.

        Args:
            series1: The first series (e.g., indicator output).
            operator: The crossing operator ('crosses_above', 'crosses_below').
            series2_or_const: The second series or a constant numerical value
                              to compare against.

        Returns:
            True if the crossing condition occurred in the latest bar, False otherwise.
        """
        if series1 is None or series2_or_const is None:
            return False

        # Ensure series1 is array-like and has at least 2 elements for crossing check
        try:
            s1_arr = np.asarray(series1).astype(float) # Ensure float type for comparison
            if s1_arr.size < 2: return False # Need at least two points to cross
        except (ValueError, TypeError) as e:
             print(f"Warning: Could not convert series1 to float array for crossing check: {e}")
             return False

        # Handle the second operand (series or constant)
        s2_arr = None
        if isinstance(series2_or_const, (int, float)):
            # Create a constant array of the same length as s1_arr
            try:
                 s2_arr = np.full_like(s1_arr, float(series2_or_const))
            except Exception as e:
                 print(f"Warning: Could not create constant array for crossing check: {e}")
                 return False
        else:
            try:
                s2_arr = np.asarray(series2_or_const).astype(float) # Ensure float type
                if s2_arr.size == 0: return False
                # Ensure compatible shapes if both are series
                # backtesting.py cross handles broadcasting for scalars, but check length
                if s1_arr.shape != s2_arr.shape:
                    # If shapes differ but lengths match, it might be okay (e.g., different initial NaNs)
                    # but comparing different length series is problematic.
                    if len(s1_arr) != len(s2_arr):
                         print(f"Warning: Crossing check between series of different lengths ({len(s1_arr)} vs {len(s2_arr)}). Returning False.")
                         return False
                    # If lengths match but shapes differ (e.g. (N,) vs (N,1)), try reshaping s2
                    try:
                         s2_arr = s2_arr.reshape(s1_arr.shape)
                    except ValueError:
                         print(f"Warning: Could not reshape series2 {s2_arr.shape} to match series1 {s1_arr.shape} for crossing check. Returning False.")
                         return False
            except (ValueError, TypeError) as e:
                 print(f"Warning: Could not convert series2 to float array for crossing check: {e}")
                 return False

        if s2_arr is None: # Should not happen if logic above is correct
             return False

        # Perform the crossing check using backtesting.lib.cross
        try:
             # Ignore RuntimeWarnings about NaNs during comparison, cross handles them
             with np.errstate(invalid='ignore'):
                 if operator == 'crosses_above':
                      # Checks if s1 crossed above s2 in the most recent step
                      return cross(s1_arr, s2_arr)
                 elif operator == 'crosses_below':
                      # Checks if s1 crossed below s2 (equivalent to s2 crossing above s1)
                      return cross(s2_arr, s1_arr)
                 else:
                      print(f"Warning: Operator '{operator}' not supported for crossing check. Use 'crosses_above' or 'crosses_below'.")
                      return False
        except Exception as e:
            # Catch potential errors within the cross function itself
            print(f"Error during backtesting.lib.cross execution: {e}")
            return False

    # --------------------------------------------------------------------------
    # Strategy Lifecycle Methods
    # --------------------------------------------------------------------------

    def init(self):
        """Initializes the strategy before the backtest starts.

        Sorts the provided `chromosome_genes` into categories (indicators,
        rules, etc.). Initializes all indicators defined by 'indicator' genes
        using `self.I()`. Initializes compliance tracking state and loads
        compliance rule values. Initializes trailing stop state.
        """
        print("Initializing GenericDNAStrategy...")
        # Clear previous state if any (e.g., during optimization runs)
        self._initialized_indicators = {}
        self._indicator_specs_by_key = {}
        self.indicator_genes = []
        self.entry_rule_genes = []
        self.exit_rule_genes = []
        self.risk_management_genes = []
        self.order_management_genes = []
        self.time_filter_genes = []
        self.trade_management_genes = []

        # Placeholder lists for different gene types
        gene_types = [
            self.indicator_genes,
            self.entry_rule_genes,
            self.exit_rule_genes,
            self.risk_management_genes,
            self.order_management_genes,
            self.time_filter_genes,
            self.trade_management_genes
        ]
        gene_type_map = {
            "indicator": self.indicator_genes,
            "entry_rule": self.entry_rule_genes,
            "exit_rule": self.exit_rule_genes,
            "risk_management": self.risk_management_genes,
            "order_management": self.order_management_genes,
            "time_filter": self.time_filter_genes,
            "trade_management": self.trade_management_genes
        }

        # Ensure chromosome_genes is a list
        if not isinstance(self.chromosome_genes, list):
             print("Error: chromosome_genes is not a list. Cannot initialize strategy.")
             return

        # --- Sort Genes by Type --- 
        for gene in self.chromosome_genes:
             # TODO: Potentially use gene.expression_level here to filter inactive genes
             # if gene.expression_level < THRESHOLD: continue
             gene_list = gene_type_map.get(gene.gene_type)
             if gene_list is not None:
                  gene_list.append(gene)
             else:
                  print(f"Warning: Unknown gene type '{gene.gene_type}' encountered.")

        # --- Initialize Indicators defined by Genes --- 
        print(f"Found {len(self.indicator_genes)} indicator genes.")
        for i, gene in enumerate(self.indicator_genes):
             params = gene.parameters
             indicator_name = params.get('indicator_name')
             period = params.get('period') # Needed for most indicators
             source_str = params.get('source', 'close') # Default to close
             
             # Generate a unique key for this indicator instance using its parameters
             param_key_string = self._get_param_key_string(params)
             indicator_key = f"ind_{i}_{indicator_name}({param_key_string})" # Unique name for self.I
             
             indicator_func = INDICATOR_MAPPING.get(indicator_name)
             if indicator_func:
                 source_data = SOURCE_MAPPING.get(source_str, lambda data: data.Close)(self.data)
                 
                 try:
                     initialized_indicator = None
                     # --- Handle specific indicator parameters --- 
                     # Generic handling first
                     if indicator_name in ["SMA", "EMA", "RSI", "ATR"]:
                         if period is None: raise ValueError("Period missing")
                         initialized_indicator = self.I(indicator_func, source_data, period, name=indicator_key)
                         print(f"  Initialized {indicator_key}")
                     elif indicator_name == "MACD":
                         fast_p = params.get('fast_period')
                         slow_p = params.get('slow_period')
                         signal_p = params.get('signal_period')
                         if None in [fast_p, slow_p, signal_p]: raise ValueError("MACD periods missing")
                         # Use custom function helper passed to self.I
                         # Name the indicator using the generated key
                         initialized_indicator = self.I(macd_func, source_data, fast=fast_p, slow=slow_p, signal=signal_p, name=indicator_key)
                         print(f"  Initialized {indicator_key} (MACD Line, Signal Line)")
                     elif indicator_name == "BBANDS":
                         std_dev = params.get('std_dev')
                         if None in [period, std_dev]: raise ValueError("BBANDS period/stddev missing")
                         # Use custom function helper passed to self.I
                         initialized_indicator = self.I(bbands_func, source_data, period=period, std_dev=std_dev, name=indicator_key)
                         print(f"  Initialized {indicator_key} (Upper, Middle, Lower Bands)")
                     else:
                          print(f"Warning: Initialization logic missing for indicator: {indicator_name}")
                          
                     # Store the initialized indicator and its parameters
                     if initialized_indicator is not None:
                         self._initialized_indicators[indicator_key] = initialized_indicator
                         self._indicator_specs_by_key[indicator_key] = params
                         
                 except Exception as e:
                     print(f"Error initializing indicator {indicator_key} from gene {params}: {e}")
             else:
                 print(f"Warning: Indicator '{indicator_name}' not found in INDICATOR_MAPPING.")

        # --- Initialize ATR if needed by Risk Management ---
        self.atr_indicator = None
        atr_needed = False
        atr_period = 14 # Default ATR period
        if self.risk_management_genes:
            # Check first risk gene if it uses ATR
            rm_params = self.risk_management_genes[0].parameters
            sl_mode = rm_params.get('stop_loss_mode')
            tp_mode = rm_params.get('take_profit_mode')
            if sl_mode == 'atr' or tp_mode == 'atr':
                atr_needed = True
                atr_period = rm_params.get('atr_period', 14) # Get period from gene or default
        
        if atr_needed:
             try:
                  # ATR requires High, Low, Close data
                  # Define a key for ATR
                  atr_key = f"ATR(period={atr_period})"
                  # Assuming self.data contains these columns
                  self.atr_indicator = self.I(ATR, self.data, atr_period, name=atr_key)
                  # Store ATR like other indicators for consistency
                  # Make sure to include 'indicator_name' in the spec for lookup
                  atr_spec = {'indicator_name': 'ATR', 'period': atr_period}
                  self._initialized_indicators[atr_key] = self.atr_indicator
                  self._indicator_specs_by_key[atr_key] = atr_spec
                  print(f"  Initialized {atr_key}")
             except Exception as e:
                  print(f"Error initializing ATR indicator: {e}")
                  self.atr_indicator = None # Ensure it's None if init fails

        # --- Initialize Compliance Tracking State --- 
        self.peak_equity = self.equity
        self.daily_pnl = 0.0 # Track PnL within the current day
        self.daily_start_equity = self.equity # Equity at start of day
        self.last_day = None
        self.hard_violation_occurred = False
        self.violation_details = {'daily_loss_limit': False, 'max_trailing_drawdown': False}
        self.contracts_allowed = 1 # Start with base contracts
        self.initial_capital = self.equity # Store initial capital for scaling baseline
        # Load compliance rules for easier access
        self.daily_loss_limit_value = abs(self.compliance_rules.get('daily_loss_limit', float('inf')))
        self.max_trailing_dd_limit_value = abs(self.compliance_rules.get('max_trailing_drawdown', float('inf')))
        hours_conf = self.compliance_rules.get('allowed_trading_hours', {})
        try:
             self.allowed_start_time = datetime.strptime(hours_conf.get('start','00:00'), '%H:%M').time()
             self.allowed_end_time = datetime.strptime(hours_conf.get('end','23:59'), '%H:%M').time()
        except ValueError:
             self.allowed_start_time = time(0,0)
             self.allowed_end_time = time(23,59)
        # Scaling plan rules
        self.scaling_plan = self.compliance_rules.get('scaling_plan', {})
        self.scaling_enabled = self.scaling_plan.get('enabled', False)
        self.scaling_levels = sorted(self.scaling_plan.get('levels', []), key=lambda x: x.get('threshold', 0))
        if self.scaling_enabled and self.scaling_levels:
            self.contracts_allowed = self.scaling_levels[0].get('contracts', 1)
            
        # State for trailing stops
        self._trailing_stop_activated = False
        self._trailing_stop_price = None
        self._entry_price = None # Track entry price for trailing stop calculation
        self._position_high = None # Track high since entry (for long)
        self._position_low = None # Track low since entry (for short)

        # Clear state on re-initialization
        self._reset_trailing_stop_state()

class BacktestingPyAdapter(AbstractBacktestEngine):
    """Adapter for using the backtesting.py library."""
    
    def run(self, chromosome: Chromosome, compliance_rules: Dict[str, Any]) -> BacktestingResults:
        """Runs a backtest using backtesting.py"""
        print("Running backtest using Backtesting.py Adapter...")
        results = BacktestingResults()
        
        # 1. Prepare Data (Ensure OHLC format)
        # backtesting.py expects columns: Open, High, Low, Close, [Volume]
        # Assuming self.data is already in this format from setup_backtester
        if not all(col in self.data.columns for col in ['Open', 'High', 'Low', 'Close']):
            raise ValueError("Data must contain Open, High, Low, Close columns for backtesting.py")
            
        # 2. Decode Chromosome / Prepare Strategy Parameters
        # Pass the *list* of genes directly to the strategy param
        strategy_params = {
             'chromosome_genes': chromosome.genes, 
             'compliance_rules': compliance_rules
        }
        
        # 3. Setup Backtest
        initial_capital = self.config.get('initial_capital', 10000)
        commission = self.config.get('commission_pct', 0.0) / 100.0 # Assuming commission is pct
        # Slippage Configuration
        slippage = self.config.get('slippage_ticks', 0) * getattr(self.data._broker, '_pip', 0.0001) # Simple fixed slippage
        
        # Margin / Futures Configuration (Example)
        # margin = self.config.get('margin_fraction', 1.0) # 1.0 for cash, < 1.0 for margin
        # trade_on_close = self.config.get('trade_on_close', True)
        # hedging = self.config.get('allow_hedging', False)
        # exclusive_orders = self.config.get('exclusive_orders', True)
        
        bt = Backtest(
            self.data, 
            GenericDNAStrategy, 
            cash=initial_capital, 
            commission=commission,
            trade_on_close=True, # Assume trading on close price, adjust if needed
            margin=self.config.get('margin_fraction', 1.0), # Use config or default
            slippage=slippage,
            exclusive_orders=self.config.get('exclusive_orders', True) # Use config or default
        )
        
        # 4. Run Backtest
        try:
            stats = bt.run(**strategy_params) # Pass params to the strategy class
            
            # 5. Extract Results
            results.stats = stats.to_dict() # Convert Series/dict to plain dict
            results.trades = stats['_trades']
            results.equity_curve = stats['_equity_curve']['Equity']
            
            # Try to get violation status from the strategy instance after run
            # This requires access to the strategy object, which Backtest stores
            if hasattr(bt, '_strategy') and bt._strategy:
                 strat_instance = bt._strategy
                 results.hard_rule_violation = getattr(strat_instance, 'hard_violation_occurred', False)
                 results.violations = getattr(strat_instance, 'violation_details', results.violations)
            else:
                 # Fallback if strategy instance not accessible
                 results.hard_rule_violation = False 
            
        except Exception as e:
            print(f"Backtest failed: {e}")
            # Assume failure means rule violation for simplicity? Or return specific error state?
            results.hard_rule_violation = True
            results.stats = {'Error': str(e)} # Store error
            # Leave trades/equity as None
            
        return results

def setup_backtester(data_path: str, config: Dict[str, Any] = None, engine_type: str = "placeholder") -> AbstractBacktestEngine:
    """Factory function to create and setup a backtest engine."""
    # TODO: Implement data loading based on data_path
    print(f"Placeholder: Loading data from {data_path}. Replace with actual loading logic.")
    try:
        # Example: Assuming CSV with Date,Open,High,Low,Close,Volume
        data = pd.read_csv(data_path, index_col='Date', parse_dates=True)
        # Ensure required columns are present and named correctly for backtesting.py
        required_cols = {'Open', 'High', 'Low', 'Close'}
        if not required_cols.issubset(data.columns):
             # Attempt to rename common alternatives if available
             rename_map = {}
             if 'open' in data.columns: rename_map['open'] = 'Open'
             if 'high' in data.columns: rename_map['high'] = 'High'
             if 'low' in data.columns: rename_map['low'] = 'Low'
             if 'close' in data.columns: rename_map['close'] = 'Close'
             if 'volume' in data.columns: rename_map['volume'] = 'Volume' # Optional
             print(f"Attempting to rename columns: {rename_map}")
             data = data.rename(columns=rename_map)
             
             if not required_cols.issubset(data.columns):
                  raise ValueError(f"Data must contain Open, High, Low, Close columns. Found: {data.columns}")
        
        # Ensure Volume exists, add dummy if not (backtesting.py needs it sometimes)
        if 'Volume' not in data.columns:
             print("Warning: No 'Volume' column found. Adding dummy volume.")
             data['Volume'] = 0

        print(f"Loaded data with shape: {data.shape}")
        
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}. Using dummy data.")
        # Fallback to dummy data if file not found
        data = pd.DataFrame({
             'Open': np.random.uniform(100, 110, size=500),
             'High': lambda df: df['Open'] + np.random.uniform(0, 2, size=500),
             'Low': lambda df: df['Open'] - np.random.uniform(0, 2, size=500),
             'Close': lambda df: df['Open'] + np.random.uniform(-1, 1, size=500),
             'Volume': np.random.randint(1000, 5000, size=500)
        }, index=pd.date_range(start='2023-01-01', periods=500, freq='D'))
        data['High'] = data[['High', 'Close']].max(axis=1) # Ensure High >= Close
        data['Low'] = data[['Low', 'Close']].min(axis=1)   # Ensure Low <= Close
        data['High'] = data[['High', 'Open']].max(axis=1) # Ensure High >= Open
        data['Low'] = data[['Low', 'Open']].min(axis=1)   # Ensure Low <= Open

    except Exception as e:
         print(f"Error loading data from {data_path}: {e}. Using dummy data.")
         # Fallback to dummy data on other errors
         data = pd.DataFrame({
             'Open': [100, 101, 102, 103, 104],
             'High': [101, 102, 103, 104, 105],
             'Low': [99, 100, 101, 102, 103],
             'Close': [101, 102, 101, 103, 104],
             'Volume': [1000, 1100, 1050, 1200, 1150]
         }, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']))
    # --- End Placeholder Data Loading ---

    if config is None:
        config = {} # Default config

    if engine_type == "placeholder":
        return PlaceholderBacktestEngine(data=data, config=config)
    elif engine_type == "backtesting.py":
        print("Using backtesting.py engine adapter.")
        # Ensure data has datetime index for backtesting.py
        if not isinstance(data.index, pd.DatetimeIndex):
             print("Warning: Dummy data index is not DatetimeIndex. Attempting conversion.")
             try:
                 data.index = pd.to_datetime(data.index)
             except Exception:
                  raise ValueError("Data index must be DatetimeIndex for backtesting.py")
        return BacktestingPyAdapter(data=data, config=config)
    # elif engine_type == "vectorbt":
        # return VectorbtEngineAdapter(data=data, config=config)
    else:
        raise ValueError(f"Unsupported backtest engine type: {engine_type}") 