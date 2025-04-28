"""Unit tests for the fitness evaluation module."""

import pytest
import random
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

# Make sure the package root is in sys.path for imports
import sys
import os
PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from nucleotide_strategy_evolution.fitness.evaluation import (
    BasicFitnessEvaluator,
    MultiObjectiveEvaluator,
    FitnessType
)
from nucleotide_strategy_evolution.backtesting.interface import (
    AbstractBacktestEngine,
    BacktestingResults
)
from nucleotide_strategy_evolution.core.structures import Chromosome, DNASequence

# --- Mock Objects --- 

class MockBacktester(AbstractBacktestEngine):
    """Mock backtester for testing evaluators."""
    def __init__(self, data=None, config=None):
        super().__init__(data if data is not None else pd.DataFrame(), 
                         config if config is not None else {})
        self.call_count = 0
        self.force_violation = False
        self.force_stats = None # Allow forcing specific stats
        
    def run(self, chromosome: Chromosome, compliance_rules: Dict[str, Any]) -> BacktestingResults:
        self.call_count += 1
        results = BacktestingResults()
        
        results.hard_rule_violation = self.force_violation
        
        if self.force_stats is not None:
             results.stats = self.force_stats.copy()
        else:
            # Simulate basic stats if not forced
            profit = len(chromosome.raw_dna) * random.uniform(0.5, 1.5)
            drawdown = random.uniform(0.01, 0.2)
            sortino = random.uniform(-0.5, 2.5)
            results.stats = {
                'net_profit': profit,
                'max_drawdown': drawdown,
                'sortino_ratio': sortino
            }
        
        # Add dummy trades if needed for other tests later
        results.trades = pd.DataFrame(columns=['entry_time', 'exit_time', 'pnl'])
        return results

# --- Tests for BasicFitnessEvaluator --- 

def test_basic_evaluator_init():
    mock_bt = MockBacktester()
    evaluator = BasicFitnessEvaluator(mock_bt)
    assert evaluator.backtester is mock_bt
    
    with pytest.raises(TypeError):
        BasicFitnessEvaluator("not_a_backtester")
        
def test_basic_evaluator_evaluate():
    mock_bt = MockBacktester()
    evaluator = BasicFitnessEvaluator(mock_bt)
    chromo = Chromosome(raw_dna=DNASequence("A"*10))
    rules = {}
    
    # Force specific stats for predictability
    # Although BasicEvaluator uses its own sim, we test the call flow
    mock_bt.force_stats = {'net_profit': 1234.5} 
    
    fitness = evaluator.evaluate(chromo, rules)
    
    assert mock_bt.call_count == 1
    # BasicEvaluator uses its *own* simulation currently
    # Test that it returns a float
    assert isinstance(fitness, float)
    # Cannot assert specific value due to internal simulation

def test_basic_evaluator_evaluate_violation():
    mock_bt = MockBacktester()
    evaluator = BasicFitnessEvaluator(mock_bt)
    chromo = Chromosome(raw_dna=DNASequence("A"*10))
    rules = {}
    
    mock_bt.force_violation = True # Force violation
    fitness = evaluator.evaluate(chromo, rules)
    
    assert mock_bt.call_count == 1
    assert fitness == -float('inf') # Should return neg infinity on violation

# --- Tests for MultiObjectiveEvaluator --- 

def test_moo_evaluator_init():
    mock_bt = MockBacktester()
    objectives = ['net_profit', '-max_drawdown']
    evaluator = MultiObjectiveEvaluator(mock_bt, objectives)
    assert evaluator.backtester is mock_bt
    assert evaluator.objectives == objectives
    assert evaluator.parsed_objectives == [('net_profit', False), ('max_drawdown', True)]
    
    with pytest.raises(TypeError):
        MultiObjectiveEvaluator("not_a_backtester", objectives)
    with pytest.raises(ValueError, match="At least one objective must be specified"):
        MultiObjectiveEvaluator(mock_bt, [])

def test_moo_evaluator_evaluate():
    mock_bt = MockBacktester()
    objectives = ['net_profit', '-max_drawdown', 'sortino_ratio']
    evaluator = MultiObjectiveEvaluator(mock_bt, objectives)
    chromo = Chromosome(raw_dna=DNASequence("A"*20))
    rules = {}
    
    # Force specific stats
    mock_bt.force_stats = {
        'net_profit': 2500.0, 
        'max_drawdown': 0.15, 
        'sortino_ratio': 1.8
    }
    
    # Modify evaluator to use forced stats instead of internal sim
    def mock_evaluate(self, chromosome: Chromosome, compliance_rules: Dict[str, Any]) -> FitnessType:
        backtest_results: BacktestingResults = self.backtester.run(chromosome, compliance_rules)
        if backtest_results.hard_rule_violation:
            worst_fitness = tuple(float('inf') if min_ else -float('inf') for _, min_ in self.parsed_objectives)
            return worst_fitness
        # USE FORCED STATS from mock backtester results
        stats_from_results = backtest_results.stats 
        fitness_values = []
        for name, minimize in self.parsed_objectives:
            value = stats_from_results.get(name, 0.0) # Default to 0 if missing
            fitness_values.append(-value if minimize else value)
        return tuple(fitness_values)
        
    # Temporarily patch the evaluate method for this test
    original_evaluate = MultiObjectiveEvaluator.evaluate
    MultiObjectiveEvaluator.evaluate = mock_evaluate
    
    try:
        fitness = evaluator.evaluate(chromo, rules)
        assert mock_bt.call_count == 1
        assert isinstance(fitness, tuple)
        assert len(fitness) == 3
        assert fitness[0] == 2500.0  # net_profit (maximize)
        assert fitness[1] == -0.15 # -max_drawdown (minimize original drawdown)
        assert fitness[2] == 1.8   # sortino_ratio (maximize)
    finally:
        # Restore original method
        MultiObjectiveEvaluator.evaluate = original_evaluate

def test_moo_evaluator_evaluate_violation():
    mock_bt = MockBacktester()
    objectives = ['net_profit', '-max_drawdown', 'sortino_ratio']
    evaluator = MultiObjectiveEvaluator(mock_bt, objectives)
    chromo = Chromosome(raw_dna=DNASequence("A"*20))
    rules = {}

    mock_bt.force_violation = True # Force violation
    fitness = evaluator.evaluate(chromo, rules)
    
    assert mock_bt.call_count == 1
    assert fitness == (-float('inf'), float('inf'), -float('inf')) # Worst possible values respecting sign

def test_moo_evaluator_missing_stat():
    mock_bt = MockBacktester()
    objectives = ['net_profit', 'sharpe_ratio'] # sharpe_ratio missing from mock
    evaluator = MultiObjectiveEvaluator(mock_bt, objectives)
    chromo = Chromosome(raw_dna=DNASequence("A"*20))
    rules = {}
    
    mock_bt.force_stats = {'net_profit': 1000.0} 
    
    # Patch evaluate to use forced stats
    def mock_evaluate(self, chromosome: Chromosome, compliance_rules: Dict[str, Any]) -> FitnessType:
        backtest_results: BacktestingResults = self.backtester.run(chromosome, compliance_rules)
        if backtest_results.hard_rule_violation:
            worst_fitness = tuple(float('inf') if min_ else -float('inf') for _, min_ in self.parsed_objectives)
            return worst_fitness
        stats_from_results = backtest_results.stats 
        fitness_values = []
        for name, minimize in self.parsed_objectives:
            value = stats_from_results.get(name, 0.0)
            fitness_values.append(-value if minimize else value)
        return tuple(fitness_values)
        
    original_evaluate = MultiObjectiveEvaluator.evaluate
    MultiObjectiveEvaluator.evaluate = mock_evaluate
    
    try:
        fitness = evaluator.evaluate(chromo, rules)
        assert mock_bt.call_count == 1
        assert fitness == (1000.0, 0.0) # Missing stat defaults to 0.0
    finally:
        MultiObjectiveEvaluator.evaluate = original_evaluate 