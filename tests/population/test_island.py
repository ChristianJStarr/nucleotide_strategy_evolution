"""Tests for the Island Model."""

import pytest
import random
from unittest.mock import MagicMock, patch

# Make imports work
import sys
import os
PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from nucleotide_strategy_evolution.population.island import IslandModel
from nucleotide_strategy_evolution.population.population import Population
from nucleotide_strategy_evolution.core.structures import Chromosome, DNASequence
from nucleotide_strategy_evolution.fitness.evaluation import FitnessType

# --- Mock Population for easier testing --- 

class MockPopulation(Population):
    def __init__(self, size: int, dna_length: int = 30, island_id: int = 0):
        super().__init__(size, dna_length)
        # Simple initialization for testing
        self.island_id = island_id
        self.individuals = [Chromosome(raw_dna=DNASequence(f"DNA_{island_id}_{j}")) for j in range(size)]
        self.fitness_scores = {j: None for j in range(size)}

    def initialize(self, strategy_template=None):
        # Override to prevent complex initialization during tests
        pass 

    def evaluate_fitnesses(self, evaluator, rules):
        # Simple mock evaluation for migration tests
        for i in range(len(self)):
             if self.fitness_scores[i] is None: # Only eval if needed
                # Assign simple fitness based on index (lower index is worse)
                 self.fitness_scores[i] = (float(i) * 10,) # Single objective tuple

# --- Test Initialization ---

@patch('nucleotide_strategy_evolution.population.island.Population', MockPopulation) # Use MockPopulation
def test_island_model_init():
    island_conf = {'num_islands': 3, 'migration_frequency': 5, 'num_migrants': 2}
    global_conf = {'population': {'size': 10, 'dna_length': 50}}
    
    model = IslandModel(island_conf, global_conf)
    
    assert model.num_islands == 3
    assert model.migration_frequency == 5
    assert model.num_migrants == 2
    assert len(model.islands) == 3
    assert all(isinstance(island, MockPopulation) for island in model.islands)
    assert all(len(island) == 10 for island in model.islands)
    assert all(island.dna_length == 50 for island in model.islands)

@patch('nucleotide_strategy_evolution.population.island.Population', MockPopulation)
def test_island_model_init_adjust_migrants():
    # num_migrants >= pop_size
    island_conf = {'num_islands': 2, 'num_migrants': 10}
    global_conf = {'population': {'size': 10}}
    model = IslandModel(island_conf, global_conf)
    assert model.num_migrants == 9 # Should be adjusted to pop_size - 1

@pytest.mark.parametrize("key, value", [
    ('num_islands', 0),
    ('migration_frequency', 0),
    ('num_migrants', -1)
])
def test_island_model_init_invalid_config(key, value):
     island_conf = {'num_islands': 3, 'migration_frequency': 5, 'num_migrants': 2}
     global_conf = {'population': {'size': 10}}
     island_conf[key] = value # Set invalid value
     with pytest.raises(ValueError):
          IslandModel(island_conf, global_conf)

# --- Test Migration --- 

@patch('nucleotide_strategy_evolution.population.island.Population', MockPopulation)
def test_migrate_best_selection():
    island_conf = {'num_islands': 2, 'num_migrants': 1, 'migration_selection_method': 'best'}
    global_conf = {'population': {'size': 5}}
    model = IslandModel(island_conf, global_conf)
    
    # Manually set fitness: Island 0 best=4, Island 1 best=4
    # Island 0: fitness (0, 10, 20, 30, 40), Best index = 4
    # Island 1: fitness (0, 10, 20, 30, 40), Best index = 4
    for i, island in enumerate(model.islands):
         island.evaluate_fitnesses(None, None) # Use mock eval
         
    migrant_island0_before = model.islands[0].individuals[4]
    migrant_island1_before = model.islands[1].individuals[4]
    worst_ind_island0_before = model.islands[0].individuals[0] # Index 0 has worst fitness 0.0
    worst_ind_island1_before = model.islands[1].individuals[0]
    
    model.migrate()
    
    # Island 0 should receive best from Island 1 (ind 4), replacing its worst (ind 0)
    assert model.islands[0].individuals[0] is migrant_island1_before
    assert model.islands[0].get_fitness(0) is None # Fitness should be reset
    assert model.islands[0].individuals[4] is migrant_island0_before # Best should remain
    
    # Island 1 should receive best from Island 0 (ind 4), replacing its worst (ind 0)
    assert model.islands[1].individuals[0] is migrant_island0_before
    assert model.islands[1].get_fitness(0) is None
    assert model.islands[1].individuals[4] is migrant_island1_before

@patch('nucleotide_strategy_evolution.population.island.Population', MockPopulation)
def test_migrate_random_selection():
    island_conf = {'num_islands': 2, 'num_migrants': 1, 'migration_selection_method': 'random'}
    global_conf = {'population': {'size': 5}}
    model = IslandModel(island_conf, global_conf)
    # Assign fitness so worst can be determined
    for i, island in enumerate(model.islands):
         island.evaluate_fitnesses(None, None)
         
    # Need to know which individuals were randomly selected
    random.seed(55)
    # Island 0 random choice from [0,1,2,3,4] -> e.g., 2
    # Island 1 random choice from [0,1,2,3,4] -> e.g., 0
    migrant_idx_island0 = 2 
    migrant_idx_island1 = 0
    migrant_island0 = model.islands[0].individuals[migrant_idx_island0]
    migrant_island1 = model.islands[1].individuals[migrant_idx_island1]
    worst_idx_island0 = 0 # Index 0 always worst here
    worst_idx_island1 = 0
    
    # Patch random.sample
    original_sample = random.sample
    mock_samples = [[migrant_idx_island0], [migrant_idx_island1]] # Define sample results
    call_count = 0
    def mock_sample_func(population, k):
        nonlocal call_count
        res = mock_samples[call_count]
        call_count += 1
        return res
        
    with patch('nucleotide_strategy_evolution.population.island.random.sample', mock_sample_func):
         model.migrate()

    # Island 0 receives migrant from Island 1 (idx 0), replaces worst (idx 0)
    assert model.islands[0].individuals[worst_idx_island0] is migrant_island1
    assert model.islands[0].get_fitness(worst_idx_island0) is None
    
    # Island 1 receives migrant from Island 0 (idx 2), replaces worst (idx 0)
    assert model.islands[1].individuals[worst_idx_island1] is migrant_island0
    assert model.islands[1].get_fitness(worst_idx_island1) is None

@patch('nucleotide_strategy_evolution.population.island.Population', MockPopulation)
def test_migrate_no_migrants():
    island_conf = {'num_islands': 2, 'num_migrants': 0}
    global_conf = {'population': {'size': 5}}
    model = IslandModel(island_conf, global_conf)
    # Get initial state
    initial_inds_0 = list(model.islands[0].individuals)
    initial_inds_1 = list(model.islands[1].individuals)
    model.migrate()
    # Populations should be unchanged
    assert model.islands[0].individuals == initial_inds_0
    assert model.islands[1].individuals == initial_inds_1

@patch('nucleotide_strategy_evolution.population.island.Population', MockPopulation)
def test_migrate_no_valid_fitness_for_best():
    island_conf = {'num_islands': 2, 'num_migrants': 1, 'migration_selection_method': 'best'}
    global_conf = {'population': {'size': 5}}
    model = IslandModel(island_conf, global_conf)
    # Keep all fitness as None
    initial_inds_0 = list(model.islands[0].individuals)
    initial_inds_1 = list(model.islands[1].individuals)
    model.migrate()
    # Migration should be skipped
    assert model.islands[0].individuals == initial_inds_0
    assert model.islands[1].individuals == initial_inds_1

# --- Test Evaluate All --- 

@patch('nucleotide_strategy_evolution.population.island.Population', MockPopulation)
def test_evaluate_all():
    island_conf = {'num_islands': 2}
    global_conf = {'population': {'size': 3}}
    model = IslandModel(island_conf, global_conf)
    
    # Use the mock evaluator defined within MockPopulation
    mock_evaluator = MagicMock() # Doesn't need specific type for this test
    mock_rules = {}
    
    # Patch the evaluate_fitnesses method on the MockPopulation instances
    for island in model.islands:
         island.evaluate_fitnesses = MagicMock()
         
    model.evaluate_all(mock_evaluator, mock_rules)
    
    # Check evaluate_fitnesses was called on each island
    assert model.islands[0].evaluate_fitnesses.call_count == 1
    assert model.islands[1].evaluate_fitnesses.call_count == 1
    model.islands[0].evaluate_fitnesses.assert_called_once_with(mock_evaluator, mock_rules)
    model.islands[1].evaluate_fitnesses.assert_called_once_with(mock_evaluator, mock_rules) 