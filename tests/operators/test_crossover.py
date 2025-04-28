"""Tests for crossover operators."""

import pytest
import random

# Make sure the package root is in sys.path for imports
import sys
import os
PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from nucleotide_strategy_evolution.operators.crossover import (
    single_point_crossover,
    uniform_crossover
)
from nucleotide_strategy_evolution.core.structures import DNASequence

# --- Test Single Point Crossover ---

def test_single_point_crossover_basic():
    p1 = DNASequence("AAAAAAAAAA") # Length 10
    p2 = DNASequence("CCCCCCCCCC")
    
    # Force crossover point for predictability
    random.seed(42) # Seed ensures random.randint(1, 9) gives a consistent value
    # With seed 42, randint(1, 9) should give 2
    expected_point = 2
    
    o1, o2 = single_point_crossover(p1, p2)
    
    assert o1.sequence == p1.sequence[:expected_point] + p2.sequence[expected_point:]
    assert o2.sequence == p2.sequence[:expected_point] + p1.sequence[expected_point:]
    assert o1.sequence == "AACCCCCCCC"
    assert o2.sequence == "CCAAAAAAAA"
    assert len(o1) == len(p1)
    assert len(o2) == len(p2)

def test_single_point_crossover_short_sequence():
    p1 = DNASequence("AC")
    p2 = DNASequence("GT")
    # Crossover point must be 1
    o1, o2 = single_point_crossover(p1, p2)
    assert o1.sequence == "AT"
    assert o2.sequence == "GC"
    
def test_single_point_crossover_no_crossover_possible():
    p1 = DNASequence("A")
    p2 = DNASequence("C")
    # Cannot perform crossover on length 1
    o1, o2 = single_point_crossover(p1, p2)
    assert o1 is p1
    assert o2 is p2

def test_single_point_crossover_different_lengths():
    p1 = DNASequence("AAAA")
    p2 = DNASequence("CCCCC")
    with pytest.raises(ValueError):
        single_point_crossover(p1, p2)

# --- Test Uniform Crossover ---

def test_uniform_crossover_no_swap():
    p1 = DNASequence("AAAAAAAAAA")
    p2 = DNASequence("CCCCCCCCCC")
    o1, o2 = uniform_crossover(p1, p2, swap_prob=0.0)
    assert o1.sequence == p1.sequence
    assert o2.sequence == p2.sequence

def test_uniform_crossover_full_swap():
    p1 = DNASequence("AAAAAAAAAA")
    p2 = DNASequence("CCCCCCCCCC")
    o1, o2 = uniform_crossover(p1, p2, swap_prob=1.0)
    # Offspring should be swapped parents
    assert o1.sequence == p2.sequence 
    assert o2.sequence == p1.sequence

def test_uniform_crossover_probability():
    p1 = DNASequence("A" * 100)
    p2 = DNASequence("C" * 100)
    swap_prob = 0.5
    num_runs = 100
    total_swaps_o1 = 0 # Count positions where o1 differs from p1
    
    for _ in range(num_runs):
        o1, o2 = uniform_crossover(p1, p2, swap_prob=swap_prob)
        for i in range(len(p1)):
            if o1.sequence[i] != p1.sequence[i]:
                total_swaps_o1 += 1
                # Check that the swap occurred correctly
                assert o1.sequence[i] == p2.sequence[i]
                assert o2.sequence[i] == p1.sequence[i]
            else:
                 assert o1.sequence[i] == p1.sequence[i]
                 assert o2.sequence[i] == p2.sequence[i]
                 
    # Expected swaps: num_runs * length * swap_prob
    expected_swaps = num_runs * len(p1) * swap_prob
    assert expected_swaps * 0.8 < total_swaps_o1 < expected_swaps * 1.2

def test_uniform_crossover_different_lengths():
    p1 = DNASequence("AAAA")
    p2 = DNASequence("CCCCC")
    with pytest.raises(ValueError):
        uniform_crossover(p1, p2)

def test_uniform_crossover_invalid_prob():
    p1 = DNASequence("AAAA")
    p2 = DNASequence("CCCC")
    with pytest.raises(ValueError):
        uniform_crossover(p1, p2, swap_prob=-0.1)
    with pytest.raises(ValueError):
        uniform_crossover(p1, p2, swap_prob=1.1) 