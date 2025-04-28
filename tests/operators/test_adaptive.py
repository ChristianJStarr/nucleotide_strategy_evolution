"""Tests for adaptive operator rate functions."""

import pytest

# Make imports work
import sys
import os
PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from nucleotide_strategy_evolution.operators.adaptive import adapt_rates

def test_adapt_rates_disabled():
    mut_configs = [{'type': 'point', 'rate': 0.1}]
    cross_config = {'type': 'single', 'rate': 0.7}
    adapt_conf = {'enabled': False}
    
    new_mut, new_cross = adapt_rates(10.0, 15.0, mut_configs, cross_config, adapt_conf)
    
    # Should return original configs if disabled
    assert new_mut is mut_configs
    assert new_cross is cross_config

def test_adapt_rates_low_diversity():
    mut_configs = [
        {'type': 'point', 'rate': 0.1},
        {'type': 'insert', 'rate': 0.05}
    ]
    cross_config = {'type': 'single', 'rate': 0.7}
    adapt_conf = {
        'enabled': True,
        'target_diversity': 15.0,
        'adjustment_factor': 0.02,
        'min_mutation_rate': 0.01, 'max_mutation_rate': 0.3,
        'min_crossover_rate': 0.5, 'max_crossover_rate': 0.9
    }
    
    current_diversity = 10.0 # Lower than target
    
    new_mut, new_cross = adapt_rates(
        current_diversity, adapt_conf['target_diversity'], 
        mut_configs, cross_config, adapt_conf
    )
    
    # Expect mutation rates to increase, crossover rate to decrease
    assert len(new_mut) == 2
    assert new_mut[0]['rate'] == pytest.approx(0.1 + 0.02)
    assert new_mut[1]['rate'] == pytest.approx(0.05 + 0.02)
    assert new_cross['rate'] == pytest.approx(0.7 - 0.02)
    # Check they are new objects (copies)
    assert new_mut is not mut_configs
    assert new_cross is not cross_config
    assert new_mut[0] is not mut_configs[0]

def test_adapt_rates_high_diversity():
    mut_configs = [
        {'type': 'point', 'rate': 0.1},
    ]
    cross_config = {'type': 'single', 'rate': 0.7}
    adapt_conf = {
        'enabled': True,
        'target_diversity': 15.0,
        'adjustment_factor': 0.02,
        'min_mutation_rate': 0.01, 'max_mutation_rate': 0.3,
        'min_crossover_rate': 0.5, 'max_crossover_rate': 0.9
    }
    
    current_diversity = 20.0 # Higher than target
    
    new_mut, new_cross = adapt_rates(
        current_diversity, adapt_conf['target_diversity'], 
        mut_configs, cross_config, adapt_conf
    )
    
    # Expect mutation rate to decrease, crossover rate to increase
    assert len(new_mut) == 1
    assert new_mut[0]['rate'] == pytest.approx(0.1 - 0.02)
    assert new_cross['rate'] == pytest.approx(0.7 + 0.02)

def test_adapt_rates_clamping():
    adapt_conf = {
        'enabled': True,
        'target_diversity': 15.0,
        'adjustment_factor': 0.1, # Large adjustment factor
        'min_mutation_rate': 0.05, 'max_mutation_rate': 0.2,
        'min_crossover_rate': 0.6, 'max_crossover_rate': 0.8
    }
    
    # --- Test hitting lower bounds --- 
    mut_low = [{'type': 'point', 'rate': 0.08}] # Start slightly above min
    cross_high = {'type': 'single', 'rate': 0.85} # Start above max
    # Low diversity -> increase mut, decrease cross
    new_mut_l, new_cross_l = adapt_rates(10.0, 15.0, mut_low, cross_high, adapt_conf)
    # Mut rate: 0.08 + 0.1 = 0.18 (within bounds)
    # Cross rate: 0.85 - 0.1 = 0.75 (within bounds)
    assert new_mut_l[0]['rate'] == pytest.approx(0.18)
    assert new_cross_l['rate'] == pytest.approx(0.75)
    
    # Start AT min mut, above max cross
    mut_at_min = [{'type': 'point', 'rate': 0.05}]
    cross_at_max = {'type': 'single', 'rate': 0.8}
    # Low diversity -> increase mut (should clamp), decrease cross (should clamp)
    new_mut_l2, new_cross_l2 = adapt_rates(10.0, 15.0, mut_at_min, cross_at_max, adapt_conf)
    assert new_mut_l2[0]['rate'] == pytest.approx(0.05 + 0.1) # 0.15
    assert new_cross_l2['rate'] == pytest.approx(0.8 - 0.1) # 0.7
    
    # --- Test hitting upper bounds --- 
    mut_high = [{'type': 'point', 'rate': 0.18}] # Start slightly below max
    cross_low = {'type': 'single', 'rate': 0.55} # Start below min
    # High diversity -> decrease mut, increase cross
    new_mut_h, new_cross_h = adapt_rates(20.0, 15.0, mut_high, cross_low, adapt_conf)
    # Mut rate: 0.18 - 0.1 = 0.08 (within bounds)
    # Cross rate: 0.55 + 0.1 = 0.65 (within bounds)
    assert new_mut_h[0]['rate'] == pytest.approx(0.08)
    assert new_cross_h['rate'] == pytest.approx(0.65)

    # Start AT max mut, below min cross
    mut_at_max = [{'type': 'point', 'rate': 0.2}]
    cross_at_min = {'type': 'single', 'rate': 0.6}
    # High diversity -> decrease mut (should clamp), increase cross (should clamp)
    new_mut_h2, new_cross_h2 = adapt_rates(20.0, 15.0, mut_at_max, cross_at_min, adapt_conf)
    assert new_mut_h2[0]['rate'] == pytest.approx(0.2 - 0.1) # 0.1
    assert new_cross_h2['rate'] == pytest.approx(0.6 + 0.1) # 0.7 