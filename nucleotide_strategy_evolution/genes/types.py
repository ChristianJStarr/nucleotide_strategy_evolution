"""Defines specific gene types and their parameter structures."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

# --- Base Gene Concept (Implicit via core.structures.Gene) ---
# We use the core.structures.Gene dataclass as the fundamental representation.
# Specific types below might eventually inherit or be validated against schemas.

# --- Placeholder Gene Type Definitions (Phase 1) ---
# These are conceptual examples; the actual implementation might use schemas
# or more detailed dataclasses validated within the decode_gene function or a factory.

@dataclass
class IndicatorGeneParams:
    """Example parameters for a technical indicator gene."""
    indicator_name: str = "SMA" # e.g., SMA, EMA, RSI, MACD
    period: int = 14
    source: str = "close"
    # ... other indicator-specific params

@dataclass
class RiskManagementGeneParams:
    """Example parameters for a risk management gene."""
    stop_loss_mode: Optional[str] = None # e.g., "ticks", "atr", "percentage"
    stop_loss_value: Optional[float] = None
    take_profit_mode: Optional[str] = None # e.g., "ticks", "atr", "rr_ratio"
    take_profit_value: Optional[float] = None
    # ... trailing stop params, position sizing logic

@dataclass
class RuleConditionParams:
    """Example parameters for a single condition within a rule."""
    indicator_gene_ref: Optional[str] = None # Reference to an IndicatorGene (or its params directly)
    indicator_name: str = "SMA"
    indicator_params: Dict[str, Any] = field(default_factory=dict) # e.g., {"period": 5}
    operator: str = ">" # e.g., >, <, ==, crosses_above, crosses_below
    threshold_type: str = "constant" # e.g., constant, indicator
    threshold_value: Any = 0 # Could be number, or another indicator ref/params
    timeframe: Optional[str] = None # e.g., "5m", "1h", "1D"

@dataclass
class EntryRuleGeneParams:
    """Example parameters for an entry rule gene."""
    conditions: List[RuleConditionParams] = field(default_factory=list)
    logic: str = "AND" # How conditions are combined (AND, OR)

@dataclass
class ExitRuleGeneParams:
    """Example parameters for an exit rule gene."""
    conditions: List[RuleConditionParams] = field(default_factory=list)
    logic: str = "AND"

# --- Mapping Gene Type Strings to Parameter Structures (Optional) ---
# This could be used for validation within the factory or decoding process
GENE_TYPE_PARAM_MAP = {
    "indicator": IndicatorGeneParams,
    "risk_management": RiskManagementGeneParams,
    "entry_rule": EntryRuleGeneParams,
    "exit_rule": ExitRuleGeneParams,
    # Add other gene types as they are defined
} 