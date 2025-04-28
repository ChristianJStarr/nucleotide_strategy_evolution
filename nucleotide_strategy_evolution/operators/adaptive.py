"""Functions for adapting genetic operator rates dynamically."""

from typing import Dict, Any, Tuple, List, MutableMapping
import copy

def adapt_rates(
    current_diversity: float, 
    target_diversity: float,
    current_mutation_configs: List[MutableMapping[str, Any]], # MutableMapping to allow modification
    current_crossover_config: MutableMapping[str, Any],
    adapt_config: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Adjusts mutation and crossover rates based on diversity.

    Args:
        current_diversity: The measured diversity metric (e.g., avg Hamming distance).
        target_diversity: The desired level of diversity.
        current_mutation_configs: List of current mutation operator config dicts (will be modified).
        current_crossover_config: Current crossover operator config dict (will be modified).
        adapt_config: Configuration for the adaptation process (adjustment_factor, min/max rates).

    Returns:
        A tuple containing the (potentially modified) list of mutation configs 
        and the crossover config dict.
    """
    if not adapt_config.get("enabled", False):
        return current_mutation_configs, current_crossover_config

    adj_factor = adapt_config.get("adjustment_factor", 0.01)
    min_mut = adapt_config.get("min_mutation_rate", 0.001)
    max_mut = adapt_config.get("max_mutation_rate", 0.5)
    min_cross = adapt_config.get("min_crossover_rate", 0.1)
    max_cross = adapt_config.get("max_crossover_rate", 0.9)
    
    # --- Deep copy to avoid modifying the original config dicts directly --- 
    # This might be important depending on how config is managed elsewhere
    # Although type hint is MutableMapping, let's assume modification is desired
    # If not, create deep copies here.
    new_mutation_configs = [cfg.copy() for cfg in current_mutation_configs]
    new_crossover_config = current_crossover_config.copy()
    
    # --- Adaptation Logic --- 
    # Simple rule: If below target, increase mutation, decrease crossover.
    #              If above target, decrease mutation, increase crossover.
    if current_diversity < target_diversity:
        mut_adjust = adj_factor
        cross_adjust = -adj_factor
        print(f"  Adapting rates: Diversity low ({current_diversity:.2f} < {target_diversity:.2f}). Increasing mutation, decreasing crossover.")
    else:
        mut_adjust = -adj_factor
        cross_adjust = adj_factor
        print(f"  Adapting rates: Diversity OK ({current_diversity:.2f} >= {target_diversity:.2f}). Decreasing mutation, increasing crossover.")

    # Adjust Mutation Rates
    for config in new_mutation_configs:
        if 'rate' in config:
            current_rate = config['rate']
            new_rate = current_rate + mut_adjust
            # Clamp within bounds
            config['rate'] = max(min_mut, min(max_mut, new_rate))
            # print(f"    Mut Rate ({config.get('type', '?')}): {current_rate:.4f} -> {config['rate']:.4f}") # Debug

    # Adjust Crossover Rate
    if 'rate' in new_crossover_config:
         current_rate = new_crossover_config['rate']
         new_rate = current_rate + cross_adjust
         new_crossover_config['rate'] = max(min_cross, min(max_cross, new_rate))
         # print(f"    Xover Rate ({new_crossover_config.get('type', '?')}): {current_rate:.4f} -> {new_crossover_config['rate']:.4f}") # Debug
         
    return new_mutation_configs, new_crossover_config

# --- Example Usage --- 
if __name__ == '__main__':
    mut_configs = [
        {'type': 'point_mutation', 'rate': 0.1},
        {'type': 'insertion', 'rate': 0.05}
    ]
    cross_config = {'type': 'single_point', 'rate': 0.7}
    adapt_conf = {
        'enabled': True,
        'target_hamming_distance': 15.0,
        'adjustment_factor': 0.02,
        'min_mutation_rate': 0.01, 'max_mutation_rate': 0.3,
        'min_crossover_rate': 0.5, 'max_crossover_rate': 0.95
    }
    
    print("--- Testing Adaptive Rates ---")
    print(f"Initial Mut Configs: {mut_configs}")
    print(f"Initial Xover Config: {cross_config}")
    
    print("\nCase 1: Diversity Low (10.0)")
    new_mut, new_cross = adapt_rates(10.0, adapt_conf['target_hamming_distance'], mut_configs, cross_config, adapt_conf)
    print(f"  New Mut Configs: {new_mut}") # Expect rates increased (0.12, 0.07)
    print(f"  New Xover Config: {new_cross}") # Expect rate decreased (0.68)
    assert new_mut[0]['rate'] > mut_configs[0]['rate']
    assert new_cross['rate'] < cross_config['rate']

    print("\nCase 2: Diversity High (20.0)")
    # Use the previously modified configs as input for next step
    mut_configs2, cross_config2 = new_mut, new_cross 
    new_mut2, new_cross2 = adapt_rates(20.0, adapt_conf['target_hamming_distance'], mut_configs2, cross_config2, adapt_conf)
    print(f"  New Mut Configs: {new_mut2}") # Expect rates decreased (0.10, 0.05)
    print(f"  New Xover Config: {new_cross2}") # Expect rate increased (0.70)
    assert new_mut2[0]['rate'] < mut_configs2[0]['rate']
    assert new_cross2['rate'] > cross_config2['rate']
    
    print("\nCase 3: Hitting Bounds")
    mut_configs_low = [
        {'type': 'point', 'rate': adapt_conf['min_mutation_rate']}
    ]
    cross_config_high = {'type': 'uniform', 'rate': adapt_conf['max_crossover_rate']}
    # Diversity is low -> try to increase mut, decrease cross
    new_mut3, new_cross3 = adapt_rates(10.0, adapt_conf['target_hamming_distance'], mut_configs_low, cross_config_high, adapt_conf)
    print(f"  Low Diversity -> Start Low Mut/High Xover -> New Mut: {new_mut3}, New Xover: {new_cross3}")
    assert new_mut3[0]['rate'] == adapt_conf['min_mutation_rate'] # Should clamp at min
    assert new_cross3['rate'] < adapt_conf['max_crossover_rate'] # Should decrease
    
    # Diversity is high -> try to decrease mut, increase cross
    new_mut4, new_cross4 = adapt_rates(20.0, adapt_conf['target_hamming_distance'], new_mut3, new_cross3, adapt_conf)
    print(f"  High Diversity -> Start Low Mut/Low Xover -> New Mut: {new_mut4}, New Xover: {new_cross4}")
    assert new_mut4[0]['rate'] == adapt_conf['min_mutation_rate'] # Still at min
    assert new_cross4['rate'] > new_cross3['rate'] # Should increase

# </rewritten_file> 