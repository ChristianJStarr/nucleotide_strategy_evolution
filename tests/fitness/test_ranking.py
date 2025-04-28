"""Tests for fitness ranking functions (NSGA-II)."""

import pytest
import numpy as np
from typing import Dict, Tuple

# Make imports work
import sys
import os
PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from nucleotide_strategy_evolution.fitness.ranking import (
    fast_non_dominated_sort,
    calculate_crowding_distance,
    FitnessType
)

# --- Test Data --- 
# Example from a common NSGA-II paper/tutorial
# Assuming maximization of both objectives for simplicity in this example
# Note: Our implementation assumes higher values are better unless obj name starts with '-'
# But for testing sorting/crowding, raw values are used directly.
MOCK_FITNESS: Dict[int, FitnessType] = {
    0: (10, 1), 1: (8, 3), 2: (8, 3), 3: (6, 5), 4: (4, 6),
    5: (2, 7), 6: (9, 2), 7: (7, 4), 8: (5, 5.5), 9: (3, 6.5)
}
# Expected Fronts (Rank 0 is non-dominated)
# F0: [0 (10,1), 6 (9,2)] 
# F1: [1 (8,3), 2 (8,3), 7 (7,4)] # 1,2 dominate 3,4,5,7,8,9; 7 dominates 3,4,5,8,9
# F2: [3 (6,5), 8 (5, 5.5)] # 3 dominates 4,5,8,9; 8 dominates 4,5,9
# F3: [4 (4,6), 9 (3, 6.5)] # 4 dominates 5,9; 9 dominates 5
# F4: [5 (2,7)]
EXPECTED_FRONTS = [
    [0, 6],
    [1, 2, 7],
    [3, 8],
    [4, 9],
    [5]
]

# --- Test Non-Dominated Sort --- 

def test_fast_non_dominated_sort_basic():
    fronts = fast_non_dominated_sort(MOCK_FITNESS)
    # Convert to sets for easier comparison regardless of order within fronts
    result_sets = [set(f) for f in fronts]
    expected_sets = [set(f) for f in EXPECTED_FRONTS]
    assert len(result_sets) == len(expected_sets)
    for i in range(len(result_sets)):
        assert result_sets[i] == expected_sets[i]

def test_fast_non_dominated_sort_all_same_front():
    fitness = {0: (1, 5), 1: (3, 3), 2: (5, 1)} # All non-dominated
    fronts = fast_non_dominated_sort(fitness)
    assert len(fronts) == 1
    assert set(fronts[0]) == {0, 1, 2}
    
def test_fast_non_dominated_sort_dominated_line():
    fitness = {0: (1, 1), 1: (2, 2), 2: (3, 3)} # Each dominates the previous
    fronts = fast_non_dominated_sort(fitness)
    assert len(fronts) == 3
    assert fronts[0] == [2]
    assert fronts[1] == [1]
    assert fronts[2] == [0]
    
def test_fast_non_dominated_sort_duplicates():
    fitness = {0: (5, 5), 1: (5, 5), 2: (3, 3)} # Duplicates on front 0
    fronts = fast_non_dominated_sort(fitness)
    assert len(fronts) == 2
    assert set(fronts[0]) == {0, 1}
    assert fronts[1] == [2]

def test_fast_non_dominated_sort_empty():
     assert fast_non_dominated_sort({}) == []

# --- Test Crowding Distance --- 

def test_calculate_crowding_distance_basic():
    # Calculate for Front 1: [1 (8,3), 2 (8,3), 7 (7,4)]
    front1_indices = EXPECTED_FRONTS[1]
    distances = calculate_crowding_distance(MOCK_FITNESS, front1_indices)
    
    # Objective 0 values: 8, 8, 7. Sorted indices by obj 0: [7, 1, 2]
    # Objective 1 values: 3, 3, 4. Sorted indices by obj 1: [1, 2, 7]
    # Boundaries: Obj 0 range = 8-7=1. Obj 1 range = 4-3=1.
    
    # Ind 7: Boundary on obj 0 min. Boundary on obj 1 max. Distance = inf + inf = inf
    assert distances[7] == float('inf')
    
    # Ind 1: Obj 0 is max (8). Obj 1 is min (3).
    # Crowding for Obj 0: (fitness[2] - fitness[7]) / range = (8 - 7) / 1 = 1.0
    # Crowding for Obj 1: (fitness[7] - fitness[1or2]) / range = (4 - 3) / 1 = 1.0
    # Total distance for ind 1 (or 2) = 1.0 + 1.0 = 2.0
    assert distances[1] == pytest.approx(2.0)
    assert distances[2] == pytest.approx(2.0) # Same values as ind 1

def test_calculate_crowding_distance_front0():
    # Front 0: [0 (10,1), 6 (9,2)]
    front0_indices = EXPECTED_FRONTS[0]
    distances = calculate_crowding_distance(MOCK_FITNESS, front0_indices)
    # Both points are boundaries, should get infinite distance
    assert distances[0] == float('inf')
    assert distances[6] == float('inf')

def test_calculate_crowding_distance_single_point_front():
     # Front 4: [5 (2,7)]
     front4_indices = EXPECTED_FRONTS[4]
     distances = calculate_crowding_distance(MOCK_FITNESS, front4_indices)
     # Single point in front also gets infinite distance
     assert distances[5] == float('inf')

def test_calculate_crowding_distance_collinear():
    fitness = {0: (1, 1), 1: (2, 2), 2: (3, 3), 3: (4,4)} # All on one line
    # All points will be in different fronts except maybe duplicates
    front_indices = [0, 1, 2, 3] # Test as if they were on the same front
    distances = calculate_crowding_distance(fitness, front_indices)
    # Boundaries (0 and 3) should be inf
    assert distances[0] == float('inf')
    assert distances[3] == float('inf')
    # Intermediate points (1 and 2)
    # Obj 0 range: 4-1=3. Obj 1 range: 4-1=3.
    # Dist(1) = (f[2]-f[0])/r0 + (f[2]-f[0])/r1 = (3-1)/3 + (3-1)/3 = 2/3 + 2/3 = 4/3
    # Dist(2) = (f[3]-f[1])/r0 + (f[3]-f[1])/r1 = (4-2)/3 + (4-2)/3 = 2/3 + 2/3 = 4/3
    assert distances[1] == pytest.approx(4.0/3.0)
    assert distances[2] == pytest.approx(4.0/3.0)

def test_calculate_crowding_distance_empty_or_single():
     assert calculate_crowding_distance({}, []) == {}
     assert calculate_crowding_distance({0: (1,1)}, [0]) == {0: float('inf')}
     assert calculate_crowding_distance({0: (1,1), 1: (2,2)}, []) == {} # Empty front indices 