"""Functions for robustness testing of evolved strategies."""

import pandas as pd
from typing import List, Dict, Any, Tuple
import copy
import numpy as np

from ..core.structures import Chromosome, Gene
from ..backtesting.interface import AbstractBacktestEngine, BacktestingResults # Assuming this exists
# Assuming FitnessType is defined elsewhere or we focus on specific metrics
from ..fitness.ranking import FitnessType 

# Define which parameter names are typically numerical and tunable
# This might need to be configurable or more dynamic later
NUMERICAL_PARAM_NAMES = {
    'period', 'fast_period', 'slow_period', 'signal_period', 'std_dev', 
    'stop_loss_value', 'take_profit_value', 
    'threshold_value', 'threshold_period', 'offset_ticks'
    # Add others as needed
}

def _perturb_chromosome(chromosome: Chromosome, perturbation_factor: float = 0.10) -> List[Tuple[str, Any, Chromosome]]:
    """Generates perturbed versions of a chromosome.

    Creates copies with one numerical parameter slightly increased or decreased.

    Returns: 
        List of tuples: (gene_info_str, perturbed_value, perturbed_chromosome)
    """
    perturbed_chromosomes = []
    original_genes = chromosome.genes

    for gene_idx, gene in enumerate(original_genes):
        gene_type = gene.gene_type
        original_params = gene.parameters
        
        # Iterate through potentially numerical parameters
        for param_name, param_value in original_params.items():
            if param_name in NUMERICAL_PARAM_NAMES and isinstance(param_value, (int, float)) and param_value != 0:
                
                # Calculate perturbed values
                change = abs(param_value * perturbation_factor)
                # Ensure minimum change if value is very small?
                if change < 1e-6: change = perturbation_factor 
                
                perturbed_values = [
                    param_value + change,
                    param_value - change
                ]
                
                for perturbed_val in perturbed_values:
                    # Create a deep copy to avoid modifying the original or other perturbed versions
                    new_chromosome = copy.deepcopy(chromosome)
                    
                    # Modify the parameter in the copied chromosome's gene
                    try:
                        # Handle potential nested parameters if necessary (e.g., rules conditions)
                        # This assumes parameters are directly in new_chromosome.genes[gene_idx].parameters
                        # More robust access might be needed depending on gene structure.
                        if param_name in new_chromosome.genes[gene_idx].parameters:
                             # Handle int vs float consistency if needed
                             if isinstance(param_value, int):
                                 new_chromosome.genes[gene_idx].parameters[param_name] = max(0, int(round(perturbed_val))) # Ensure non-negative ints
                             else:
                                 new_chromosome.genes[gene_idx].parameters[param_name] = perturbed_val
                                 
                             gene_info = f"Gene_{gene_idx}_{gene_type}_{param_name}"
                             perturbed_chromosomes.append((gene_info, perturbed_val, new_chromosome))
                        else:
                             print(f"Warning: Parameter '{param_name}' not found in copied gene {gene_idx}. Skipping perturbation.")
                             
                    except IndexError:
                         print(f"Warning: Gene index {gene_idx} out of bounds during perturbation. Skipping.")
                    except Exception as e:
                         print(f"Warning: Error perturbing {param_name} in gene {gene_idx}: {e}")
                         
    return perturbed_chromosomes

def parameter_sensitivity_analysis(
    chromosome: Chromosome, 
    base_results: BacktestingResults, # Assume this contains baseline performance metrics
    backtest_engine: AbstractBacktestEngine, 
    perturbation_factor: float = 0.10,
    metrics_to_compare: List[str] = ['net_profit', 'max_drawdown', 'sortino_ratio'] # Example metrics
) -> Dict[str, Any]:
    """Performs parameter sensitivity analysis by slightly modifying parameters.

    Args:
        chromosome: The original evolved chromosome.
        base_results: The BacktestingResults object from the original chromosome's run.
        backtest_engine: The backtesting engine instance.
        perturbation_factor: The fractional change to apply to numerical parameters (e.g., 0.1 for 10%).
        metrics_to_compare: List of keys expected in the backtest_results.stats dictionary to compare.

    Returns:
        A dictionary summarizing the sensitivity results.
    """
    print(f"Running Parameter Sensitivity Analysis (Factor: {perturbation_factor:.1%})...")
    
    # --- Get Baseline Metrics --- 
    baseline_metrics = {}
    if not hasattr(base_results, 'stats') or not isinstance(base_results.stats, dict):
         print("Warning: base_results object does not have a valid 'stats' dictionary. Cannot get baseline.")
         return {"error": "Invalid baseline results object"}
         
    for metric in metrics_to_compare:
        baseline_metrics[metric] = base_results.stats.get(metric)
        if baseline_metrics[metric] is None:
            print(f"Warning: Baseline metric '{metric}' not found in base_results.stats.")
    
    print(f"Baseline Metrics: {baseline_metrics}")
    if not any(v is not None for v in baseline_metrics.values()):
         print("Warning: No valid baseline metrics found. Sensitivity analysis may be uninformative.")
         
    # --- Generate and Evaluate Perturbed Chromosomes --- 
    perturbed_versions = _perturb_chromosome(chromosome, perturbation_factor)
    if not perturbed_versions:
        print("No numerical parameters found or perturbed. Skipping analysis.")
        return {"parameter_sensitivity": "No numerical parameters found/perturbed"}
        
    sensitivity_results = {}
    print(f"Evaluating {len(perturbed_versions)} perturbed chromosomes...")
    
    for gene_info, perturbed_value, perturbed_chromo in perturbed_versions:
        print(f"  Testing {gene_info} = {perturbed_value:.4f}...")
        try:
            # Run backtest - assuming backtester uses default compliance rules or they are part of engine state
            # If compliance rules need to be passed, modify the call
            perturbed_results = backtest_engine.run(perturbed_chromo)
            
            result_summary = {'perturbed_value': perturbed_value, 'metrics': {}}
            if hasattr(perturbed_results, 'stats') and isinstance(perturbed_results.stats, dict):
                for metric in metrics_to_compare:
                     metric_value = perturbed_results.stats.get(metric)
                     result_summary['metrics'][metric] = metric_value
                     # Calculate percentage change from baseline if possible
                     if baseline_metrics.get(metric) is not None and metric_value is not None and baseline_metrics[metric] != 0:
                          try:
                              pct_change = (metric_value - baseline_metrics[metric]) / abs(baseline_metrics[metric]) * 100
                              result_summary['metrics'][f'{metric}_pct_change'] = pct_change
                          except ZeroDivisionError:
                              pass # Baseline was zero
            else:
                 print(f"    Warning: Perturbed results for {gene_info} lack a valid 'stats' dict.")
                 
            sensitivity_results[gene_info] = result_summary
            
        except Exception as e:
            print(f"    Error backtesting perturbed chromosome {gene_info}: {e}")
            sensitivity_results[gene_info] = {'perturbed_value': perturbed_value, 'error': str(e)}

    print("Parameter Sensitivity Analysis complete.")
    # TODO: Add more sophisticated summary/aggregation of results
    return {
        "parameter_sensitivity_details": sensitivity_results, 
        "baseline_metrics": baseline_metrics
    }

def market_regime_sensitivity(chromosome: Chromosome, backtest_engine: AbstractBacktestEngine):
    """Evaluates strategy performance across different market regimes.

    TODO: Implement logic
        - Define market regimes (e.g., using VIX levels, MA trends, etc.).
        - Split historical data based on these regimes.
        - Run backtests for the chromosome on each regime's data subset.
        - Analyze performance consistency across regimes.
    """
    print("TODO: Implement market_regime_sensitivity")
    # Example steps:
    # 1. Load full data from backtest_engine or path
    # 2. Define regime identification logic (e.g., function taking data, returning regime labels/masks)
    # 3. Split data by regime label
    # 4. Loop through regimes:
    #    a. Create temporary backtest engine with subset data?
    #    b. Run backtest on regime data subset
    #    c. Store performance metrics
    # 5. Analyze performance variation
    return {"regime_sensitivity": "Not Implemented"}

def cost_sensitivity(chromosome: Chromosome, base_results: BacktestingResults, backtest_engine: AbstractBacktestEngine):
    """Tests strategy performance with varying slippage and commission assumptions.

    TODO: Implement logic
        - Requires backtest_engine to support configurable costs.
        - Define range of slippage/commission values to test.
        - Re-run backtests with different cost settings.
        - Analyze impact on profitability.
    """
    print("TODO: Implement cost_sensitivity")
    # Example steps:
    # 1. Define cost scenarios (low, medium, high commission/slippage)
    # 2. Loop through scenarios:
    #    a. Configure backtest_engine with new cost settings (if possible)
    #    b. Run backtest
    #    c. Store results
    # 3. Compare performance degradation
    return {"cost_sensitivity": "Not Implemented"} 