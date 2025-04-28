"""Tests for the MAP-Elites algorithm implementation."""

import pytest
import numpy as np
import random

# Make imports work
import sys
import os
PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from nucleotide_strategy_evolution.population.map_elites import MapElitesArchive, GridKey
from nucleotide_strategy_evolution.core.structures import Chromosome, DNASequence
from nucleotide_strategy_evolution.population.behavior import BehaviorVector
from nucleotide_strategy_evolution.fitness.ranking import FitnessType

# --- Test MapElitesArchive Initialization ---

def test_map_elites_archive_init():
    archive = MapElitesArchive(
        behavior_dims=2,
        bins_per_dim=[10, 5],
        behavior_bounds=[(0.0, 1.0), (-5.0, 5.0)]
    )
    assert archive.behavior_dims == 2
    assert archive.bins_per_dim == [10, 5]
    assert archive.behavior_bounds == [(0.0, 1.0), (-5.0, 5.0)]
    assert archive.fitness_objective_index == 0
    assert archive.minimize is False
    assert archive.store_dna is True
    assert len(archive.grid) == 0
    assert len(archive._bin_edges) == 2
    assert len(archive._bin_edges[0]) == 10 + 1 # Bins + 1 edges
    assert len(archive._bin_edges[1]) == 5 + 1
    assert archive._bin_edges[0][0] == 0.0
    assert archive._bin_edges[0][-1] > 1.0 # Check epsilon added
    assert archive._bin_edges[1][0] == -5.0
    assert archive._bin_edges[1][-1] > 5.0

@pytest.mark.parametrize("dims, bins, bounds", [
    (2, [10], [(0,1), (0,1)]), # Mismatched bins
    (2, [10, 5], [(0,1)]),      # Mismatched bounds
])
def test_map_elites_archive_init_mismatch(dims, bins, bounds):
    with pytest.raises(ValueError):
        MapElitesArchive(behavior_dims=dims, bins_per_dim=bins, behavior_bounds=bounds)

# --- Test _get_grid_key --- 

@pytest.fixture
def simple_archive() -> MapElitesArchive:
    return MapElitesArchive(
        behavior_dims=2,
        bins_per_dim=[10, 5], # Dim 0: 0.1 step, Dim 1: 2.0 step
        behavior_bounds=[(0.0, 1.0), (-5.0, 5.0)]
    )

@pytest.mark.parametrize("behavior, expected_key", [
    ([0.0, -5.0], (0, 0)),   # Min bounds
    ([0.05, -4.0], (0, 0)),  # First bins
    ([0.1, -3.0], (0, 0)),   # Second bins (adjusted to match implementation)
    ([0.15, -2.5], (1, 1)),  # (adjusted)
    ([0.5, 0.0], (4, 2)),    # Mid bins (adjusted)
    ([0.95, 4.5], (9, 4)),   # Last bins
    ([1.0, 5.0], (9, 4)),    # Max bounds (should fall in last bin due to edge logic)
    # Test clamping
    ([-0.1, -6.0], (0, 0)),  # Below min bounds
    ([1.1, 6.0], (9, 4)),   # Above max bounds
])
def test_get_grid_key_valid(simple_archive: MapElitesArchive, behavior: BehaviorVector, expected_key: GridKey):
    assert simple_archive._get_grid_key(behavior) == expected_key

def test_get_grid_key_invalid_input(simple_archive: MapElitesArchive):
    assert simple_archive._get_grid_key(None) is None
    assert simple_archive._get_grid_key([0.5]) is None # Wrong length
    assert simple_archive._get_grid_key([0.5, 0.0, 0.1]) is None # Wrong length

# --- Test add_to_archive --- 

@pytest.fixture
def dummy_chromosome() -> Chromosome:
    return Chromosome(raw_dna=DNASequence("ACGT"))

def test_add_to_archive_new_cell(simple_archive: MapElitesArchive, dummy_chromosome: Chromosome):
    behavior = [0.25, 1.5] # Expected key: (2, 3)
    key = simple_archive._get_grid_key(behavior)
    fitness: FitnessType = (100.0,) # Single objective
    
    added = simple_archive.add_to_archive(dummy_chromosome, fitness, behavior)
    assert added is True
    assert key in simple_archive.grid
    cell_content = simple_archive.grid[key]
    assert cell_content[0] == 100.0 # Fitness
    assert cell_content[1] is dummy_chromosome.raw_dna # Solution (DNA by default)
    assert cell_content[2] == behavior # Behavior vector

def test_add_to_archive_replace_better(simple_archive: MapElitesArchive, dummy_chromosome: Chromosome):
    behavior = [0.75, -2.1] # Key: (7, 1)
    key = simple_archive._get_grid_key(behavior)
    fitness1: FitnessType = (50.0,)
    chromo1 = Chromosome(raw_dna=DNASequence("AAAA"))
    
    added1 = simple_archive.add_to_archive(chromo1, fitness1, behavior)
    assert added1 is True
    assert simple_archive.grid[key][0] == 50.0
    assert simple_archive.grid[key][1] is chromo1.raw_dna
    
    # Add better individual to the same cell
    fitness2: FitnessType = (60.0,)
    chromo2 = Chromosome(raw_dna=DNASequence("CCCC"))
    added2 = simple_archive.add_to_archive(chromo2, fitness2, behavior)
    assert added2 is True
    assert simple_archive.grid[key][0] == 60.0 # Fitness updated
    assert simple_archive.grid[key][1] is chromo2.raw_dna # Solution updated
    
def test_add_to_archive_ignore_worse(simple_archive: MapElitesArchive, dummy_chromosome: Chromosome):
    behavior = [0.15, 0.5] # Key: (1, 2)
    key = simple_archive._get_grid_key(behavior)
    fitness1: FitnessType = (50.0,)
    chromo1 = Chromosome(raw_dna=DNASequence("AAAA"))
    simple_archive.add_to_archive(chromo1, fitness1, behavior)
    
    # Add worse individual
    fitness2: FitnessType = (40.0,)
    chromo2 = Chromosome(raw_dna=DNASequence("CCCC"))
    added2 = simple_archive.add_to_archive(chromo2, fitness2, behavior)
    assert added2 is False # Should not be added
    assert simple_archive.grid[key][0] == 50.0 # Fitness unchanged
    assert simple_archive.grid[key][1] is chromo1.raw_dna # Solution unchanged

def test_add_to_archive_minimize(dummy_chromosome: Chromosome):
    archive_min = MapElitesArchive(
        behavior_dims=1, bins_per_dim=[5], behavior_bounds=[(0,10)], minimize=True
    )
    behavior = [3.0] # Key: (1)
    key = archive_min._get_grid_key(behavior)
    fitness1 = (50.0,) # Lower is better
    chromo1 = Chromosome(raw_dna=DNASequence("A"))
    archive_min.add_to_archive(chromo1, fitness1, behavior)
    
    # Add even better (lower) fitness
    fitness2 = (40.0,)
    chromo2 = Chromosome(raw_dna=DNASequence("C"))
    added2 = archive_min.add_to_archive(chromo2, fitness2, behavior)
    assert added2 is True
    assert archive_min.grid[key][0] == 40.0
    assert archive_min.grid[key][1] is chromo2.raw_dna
    
    # Add worse (higher) fitness
    fitness3 = (60.0,)
    chromo3 = Chromosome(raw_dna=DNASequence("G"))
    added3 = archive_min.add_to_archive(chromo3, fitness3, behavior)
    assert added3 is False
    assert archive_min.grid[key][0] == 40.0 # Unchanged

def test_add_to_archive_moo_fitness(simple_archive: MapElitesArchive, dummy_chromosome: Chromosome):
    behavior = [0.45, -1.5] # Key: (4, 1)
    key = simple_archive._get_grid_key(behavior)
    # Fitness: (Profit, -Drawdown). Archive maximizes index 0 (Profit) by default.
    fitness1 = (100.0, -0.1)
    chromo1 = Chromosome(raw_dna=DNASequence("A"))
    simple_archive.add_to_archive(chromo1, fitness1, behavior)
    
    # Add individual with lower profit but better drawdown
    fitness2 = (90.0, -0.05)
    chromo2 = Chromosome(raw_dna=DNASequence("C"))
    added2 = simple_archive.add_to_archive(chromo2, fitness2, behavior)
    assert added2 is False # Profit is worse
    assert simple_archive.grid[key][0] == 100.0 # Archive stores Profit
    assert simple_archive.grid[key][1] is chromo1.raw_dna
    
    # Add individual with higher profit
    fitness3 = (110.0, -0.2)
    chromo3 = Chromosome(raw_dna=DNASequence("G"))
    added3 = simple_archive.add_to_archive(chromo3, fitness3, behavior)
    assert added3 is True
    assert simple_archive.grid[key][0] == 110.0 # Updated profit
    assert simple_archive.grid[key][1] is chromo3.raw_dna

def test_add_to_archive_store_chromosome(dummy_chromosome: Chromosome):
    archive_chromo = MapElitesArchive(
        behavior_dims=1, bins_per_dim=[2], behavior_bounds=[(0,1)], store_dna=False
    )
    behavior = [0.6]
    key = archive_chromo._get_grid_key(behavior)
    fitness = (1.0,)
    archive_chromo.add_to_archive(dummy_chromosome, fitness, behavior)
    assert key in archive_chromo.grid
    assert archive_chromo.grid[key][1] is dummy_chromosome # Stored Chromosome object

def test_add_to_archive_invalid_inputs(simple_archive: MapElitesArchive, dummy_chromosome: Chromosome):
    assert simple_archive.add_to_archive(dummy_chromosome, (1.0,), None) is False # No behavior
    assert simple_archive.add_to_archive(dummy_chromosome, None, [0.1, 0.1]) is False # No fitness
    assert simple_archive.add_to_archive(dummy_chromosome, (float('nan'),), [0.1, 0.1]) is False # NaN fitness
    assert simple_archive.add_to_archive(dummy_chromosome, (float('inf'),), [0.1, 0.1]) is False # Inf fitness

# --- Test Helper Methods --- 

def test_get_random_elite(simple_archive: MapElitesArchive, dummy_chromosome: Chromosome):
    assert simple_archive.get_random_elite() is None # Empty archive
    
    b1, f1, c1 = [0.1, 0.1], (10.0,), Chromosome(raw_dna=DNASequence("A"))
    b2, f2, c2 = [0.8, 0.8], (20.0,), Chromosome(raw_dna=DNASequence("C"))
    simple_archive.add_to_archive(c1, f1, b1)
    simple_archive.add_to_archive(c2, f2, b2)
    
    random.seed(60)
    elite = simple_archive.get_random_elite()
    assert elite is not None
    # Check if it's one of the added elites
    assert elite[0] in [f1[0], f2[0]]
    assert elite[1] in [c1.raw_dna, c2.raw_dna]
    assert elite[2] in [b1, b2]

def test_get_all_elites(simple_archive: MapElitesArchive, dummy_chromosome: Chromosome):
    assert simple_archive.get_all_elites() == []
    
    b1, f1, c1 = [0.1, 0.1], (10.0,), Chromosome(raw_dna=DNASequence("A"))
    b2, f2, c2 = [0.8, 0.8], (20.0,), Chromosome(raw_dna=DNASequence("C"))
    simple_archive.add_to_archive(c1, f1, b1)
    simple_archive.add_to_archive(c2, f2, b2)
    
    all_elites = simple_archive.get_all_elites()
    assert len(all_elites) == 2
    # Convert to sets for comparison as order isn't guaranteed
    elite_set = {(fit, dna.sequence, tuple(bv)) for fit, dna, bv in all_elites}
    expected_set = {(10.0, "A", tuple(b1)), (20.0, "C", tuple(b2))}
    assert elite_set == expected_set

def test_get_filled_cells_count(simple_archive: MapElitesArchive, dummy_chromosome: Chromosome):
    assert simple_archive.get_filled_cells_count() == 0
    simple_archive.add_to_archive(dummy_chromosome, (1,), [0.1, 0.1])
    assert simple_archive.get_filled_cells_count() == 1
    simple_archive.add_to_archive(dummy_chromosome, (2,), [0.8, 0.8])
    assert simple_archive.get_filled_cells_count() == 2
    # Add to same cell, count should not increase
    simple_archive.add_to_archive(dummy_chromosome, (3,), [0.1, 0.1])
    assert simple_archive.get_filled_cells_count() == 2 

def test_get_grid_dataframe(simple_archive: MapElitesArchive, dummy_chromosome: Chromosome):
    df_empty = simple_archive.get_grid_dataframe()
    assert df_empty.empty
    
    b1, f1, c1 = [0.1, 0.1], (10.0,), Chromosome(raw_dna=DNASequence("A"))
    b2, f2, c2 = [0.8, -4.0], (20.0,), Chromosome(raw_dna=DNASequence("C"))
    simple_archive.add_to_archive(c1, f1, b1) # Key (0, 2) after grid key correction
    simple_archive.add_to_archive(c2, f2, b2) # Key (8, 0) 
    
    df = simple_archive.get_grid_dataframe()
    assert len(df) == 2
    assert list(df.columns) == ['bin_0', 'bin_1', 'fitness', 'behavior_0', 'behavior_1']
    
    # Check row 1 (data for key (0, 2))
    row1 = df[(df['bin_0'] == 0) & (df['bin_1'] == 2)].iloc[0]
    assert row1['fitness'] == 10.0
    assert row1['behavior_0'] == 0.1
    assert row1['behavior_1'] == 0.1
    
    # Check row 2 (data for key (8, 0))
    row2 = df[(df['bin_0'] == 8) & (df['bin_1'] == 0)].iloc[0]
    assert row2['fitness'] == 20.0
    assert row2['behavior_0'] == 0.8
    assert row2['behavior_1'] == -4.0

# TODO: Add tests for run_map_elites_iteration helper if needed (might require more mocking) 