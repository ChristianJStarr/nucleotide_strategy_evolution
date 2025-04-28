"""Tests for population diversity functions."""

import pytest
import math
from typing import List, Dict, Sequence, Optional

# Make imports work
import sys
import os
PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from nucleotide_strategy_evolution.core.structures import DNASequence
from nucleotide_strategy_evolution.fitness.ranking import FitnessType
from nucleotide_strategy_evolution.population.diversity import (
    calculate_hamming_distance,
    calculate_average_hamming_distance,
    sharing_function,
    apply_fitness_sharing,
    calculate_behavioral_distance,
    NoveltyArchive,
    calculate_novelty_score,
    assign_species
)
# Need BehaviorVector type hint
from nucleotide_strategy_evolution.population.behavior import BehaviorVector

# --- Test Genotypic Diversity ---

def test_calculate_hamming_distance():
    seq_a = DNASequence("ATGCATGC")
    seq_b = DNASequence("ATGGATAC") # 2 diffs
    seq_c = DNASequence("ATGCATGC") # 0 diffs
    seq_d = DNASequence("TGCATGCA") # 8 diffs
    seq_e = DNASequence("AAAAAAAA")
    seq_f = DNASequence("TTTTTTTT")
    seq_g = DNASequence("ATGC") # Different length
    
    assert calculate_hamming_distance(seq_a, seq_b) == 2
    assert calculate_hamming_distance(seq_a, seq_c) == 0
    assert calculate_hamming_distance(seq_a, seq_d) == 8
    assert calculate_hamming_distance(seq_e, seq_f) == 8
    # Test self comparison
    assert calculate_hamming_distance(seq_a, seq_a) == 0
    # Test different lengths
    with pytest.raises(ValueError, match="requires sequences of equal length"):
        calculate_hamming_distance(seq_a, seq_g)

def test_calculate_average_hamming_distance():
    seq_a = DNASequence("AAAA")
    seq_b = DNASequence("AAAT") # Dist 1
    seq_c = DNASequence("AATT") # Dist 2
    seq_d = DNASequence("ATTT") # Dist 3
    seq_e = DNASequence("TTTT") # Dist 4
    pop_dna = [seq_a, seq_b, seq_c, seq_d, seq_e]
    # Pairs (Dist): (a,b)=1, (a,c)=2, (a,d)=3, (a,e)=4 -> 10
    #               (b,c)=1, (b,d)=2, (b,e)=3 -> 6
    #               (c,d)=1, (c,e)=2 -> 3
    #               (d,e)=1 -> 1
    # Total dist = 10 + 6 + 3 + 1 = 20
    # Num pairs = 4+3+2+1 = 10
    # Expected avg = 20 / 10 = 2.0
    assert calculate_average_hamming_distance(pop_dna, sample_size=0) == pytest.approx(2.0)
    assert calculate_average_hamming_distance(pop_dna) == pytest.approx(2.0) # Default sample > num pairs

def test_calculate_average_hamming_distance_sampling():
    # Use a larger pop where sampling makes sense
    pop_dna = [DNASequence.random(20) for _ in range(30)]
    avg_dist_sample = calculate_average_hamming_distance(pop_dna, sample_size=50)
    # Can't assert exact value, but check it's a non-negative float
    assert isinstance(avg_dist_sample, float)
    assert avg_dist_sample >= 0.0

def test_calculate_average_hamming_distance_edge_cases():
    assert calculate_average_hamming_distance([]) == 0.0
    assert calculate_average_hamming_distance([DNASequence("AAA")]) == 0.0
    # Mixed lengths (should skip invalid pairs)
    pop_mixed = [DNASequence("AAAA"), DNASequence("TTTT"), DNASequence("CC")]
    # Only AAAA vs TTTT is valid (dist 4). 1 pair compared.
    assert calculate_average_hamming_distance(pop_mixed, sample_size=0) == pytest.approx(4.0)

# --- Test Fitness Sharing --- 

def test_sharing_function():
    sigma = 5.0
    alpha = 1.0
    assert sharing_function(0, sigma, alpha) == 1.0
    assert sharing_function(2, sigma, alpha) == pytest.approx(1.0 - (2/5)**1) # 0.6
    assert sharing_function(5, sigma, alpha) == 0.0 # d >= sigma
    assert sharing_function(6, sigma, alpha) == 0.0
    # Test alpha = 2
    assert sharing_function(2, sigma, alpha=2.0) == pytest.approx(1.0 - (2/5)**2) # 1 - 0.16 = 0.84
    # Test sigma = 0
    assert sharing_function(2, 0.0, alpha) == 0.0

def test_apply_fitness_sharing_single_obj():
    pop_dna = [
        DNASequence("AAAAAAAA"), # 0
        DNASequence("AAAAAAAT"), # 1 (Dist=1 to 0)
        DNASequence("TTTTTTTT"), # 2 (Dist=8 to 0, 7 to 1)
    ]
    original_fits = {0: 100.0, 1: 90.0, 2: 80.0}
    sigma = 3.0 # Share if dist < 3
    alpha = 1.0
    
    # Niche counts:
    # N(0) = sh(0,0) + sh(0,1) + sh(0,2) = 1 + sh(1) + sh(8) = 1 + (1 - 1/3) + 0 = 1 + 2/3 = 5/3
    # N(1) = sh(1,0) + sh(1,1) + sh(1,2) = sh(1) + 1 + sh(7) = (1 - 1/3) + 1 + 0 = 2/3 + 1 = 5/3
    # N(2) = sh(2,0) + sh(2,1) + sh(2,2) = sh(8) + sh(7) + 1 = 0 + 0 + 1 = 1
    niche0 = 1.0 + (1.0 - 1.0/3.0) + 0.0 
    niche1 = (1.0 - 1.0/3.0) + 1.0 + 0.0
    niche2 = 0.0 + 0.0 + 1.0
    assert niche0 == niche1 == pytest.approx(5.0/3.0)
    assert niche2 == 1.0
    
    adj_fits = apply_fitness_sharing(pop_dna, original_fits, sigma, alpha)
    
    assert adj_fits[0] == pytest.approx(100.0 / niche0) # 100 / (5/3) = 60
    assert adj_fits[1] == pytest.approx(90.0 / niche1) # 90 / (5/3) = 54
    assert adj_fits[2] == pytest.approx(80.0 / niche2) # 80 / 1 = 80

def test_apply_fitness_sharing_multi_obj():
    pop_dna = [
        DNASequence("AAAAAAAA"), # 0
        DNASequence("AAAAAAAT"), # 1 (Dist=1 to 0)
    ]
    original_fits_moo: Dict[int, FitnessType] = {
        0: (100.0, 5.0),
        1: (95.0, 4.8),
    }
    sigma = 2.0
    alpha = 1.0
    
    # Niche counts:
    # N(0) = sh(0,0) + sh(0,1) = 1 + sh(1) = 1 + (1 - 1/2) = 1.5
    # N(1) = sh(1,0) + sh(1,1) = sh(1) + 1 = (1 - 1/2) + 1 = 1.5
    niche = 1.5
    
    adj_fits = apply_fitness_sharing(pop_dna, original_fits_moo, sigma, alpha, is_multi_objective=True)
    
    assert len(adj_fits) == 2
    assert adj_fits[0][0] == pytest.approx(100.0 / niche)
    assert adj_fits[0][1] == pytest.approx(5.0 / niche)
    assert adj_fits[1][0] == pytest.approx(95.0 / niche)
    assert adj_fits[1][1] == pytest.approx(4.8 / niche)

def test_apply_fitness_sharing_mixed_lengths():
    pop_dna = [DNASequence("AAAA"), DNASequence("CCCC"), DNASequence("TT")]
    original_fits = {0: 100.0, 1: 90.0, 2: 80.0}
    sigma = 5.0
    # Niche counts should ignore comparisons between different lengths
    # N(0) = sh(0,0) + sh(0,1) + sh(0,2)=NA = 1 + sh(4) + 0 = 1 + (1-4/5) = 1.2
    # N(1) = sh(1,0) + sh(1,1) + sh(1,2)=NA = sh(4) + 1 + 0 = 1.2
    # N(2) = sh(2,0)=NA + sh(2,1)=NA + sh(2,2) = 0 + 0 + 1 = 1.0
    niche01 = 1.0 + (1.0 - 4.0/5.0)
    niche2 = 1.0
    
    adj_fits = apply_fitness_sharing(pop_dna, original_fits, sigma)
    assert adj_fits[0] == pytest.approx(100.0 / niche01)
    assert adj_fits[1] == pytest.approx(90.0 / niche01)
    assert adj_fits[2] == pytest.approx(80.0 / niche2)

# --- Test Behavioral Diversity --- 

def test_calculate_behavioral_distance():
    bv1 = [1.0, 2.0, 3.0]
    bv2 = [1.0, 2.0, 3.0] # Identical
    bv3 = [1.0, 2.0, 4.0] # Differs by 1
    bv4 = [2.0, 3.0, 4.0] # Differs by sqrt(1^2+1^2+1^2) = sqrt(3)
    bv5 = [1.0, 2.0] # Different length
    
    assert calculate_behavioral_distance(bv1, bv2) == 0.0
    assert calculate_behavioral_distance(bv1, bv3) == pytest.approx(1.0) # sqrt((1-1)^2+(2-2)^2+(3-4)^2) = sqrt(1)
    assert calculate_behavioral_distance(bv1, bv4) == pytest.approx(math.sqrt(3))
    assert calculate_behavioral_distance(bv1, bv5) == float('inf')

# --- Test Novelty Search Components ---

def test_novelty_archive():
    archive = NoveltyArchive(capacity=3)
    assert len(archive) == 0
    
    bv1 = [1.0, 1.0]
    bv2 = [2.0, 2.0]
    bv3 = [3.0, 3.0]
    bv4 = [4.0, 4.0]
    
    archive.add_to_archive(bv1)
    assert len(archive) == 1
    assert archive.get_archive_behaviors() == [bv1]
    
    archive.add_to_archive(bv2)
    archive.add_to_archive(bv3)
    assert len(archive) == 3
    assert archive.get_archive_behaviors() == [bv1, bv2, bv3]
    
    # Add one more, should remove bv1 (oldest)
    archive.add_to_archive(bv4)
    assert len(archive) == 3
    assert archive.get_archive_behaviors() == [bv2, bv3, bv4]
    
    # Test adding None
    archive.add_to_archive(None)
    assert len(archive) == 3 # Length should not change

def test_calculate_novelty_score():
    archive = NoveltyArchive(capacity=5)
    archive.add_to_archive([0.0, 0.0])
    archive.add_to_archive([10.0, 10.0])
    
    pop_bvs: List[Optional[BehaviorVector]] = [
        [1.0, 1.0], # 0
        [1.1, 1.1], # 1
        [5.0, 5.0], # 2
        [5.1, 5.0], # 3
        None,       # 4 (No behavior)
        [1.0, 1.0]  # 5 (Duplicate of 0)
    ]
    k = 3
    
    # --- Test individual 0 --- 
    # Neighbors (excluding self): pop[1], pop[2], pop[3], pop[5], arc[0], arc[1]
    # BVs: [1.1, 1.1], [5.0, 5.0], [5.1, 5.0], [1.0, 1.0], [0.0, 0.0], [10.0, 10.0]
    # Dists to [1.0, 1.0]: sqrt(0.02), sqrt(32), sqrt(32.81), 0, sqrt(2), sqrt(162)
    # Sorted Dists: 0, ~0.141, ~1.414, ~5.65, ~5.72, ~12.7
    # k=3 smallest: 0, 0.141, 1.414
    # Avg = (0 + sqrt(0.02) + sqrt(2)) / 3 
    expected_novelty0 = (0 + math.sqrt(0.02) + math.sqrt(2)) / 3
    novelty0 = calculate_novelty_score(pop_bvs[0], pop_bvs, archive, k)
    assert novelty0 == pytest.approx(expected_novelty0)
    
    # --- Test individual 2 --- 
    # Neighbors (excluding self): pop[0], pop[1], pop[3], pop[5], arc[0], arc[1]
    # BVs: [1.0, 1.0], [1.1, 1.1], [5.1, 5.0], [1.0, 1.0], [0.0, 0.0], [10.0, 10.0]
    # Dists to [5.0, 5.0]: sqrt(32), sqrt(30.41), sqrt(0.01), sqrt(32), sqrt(50), sqrt(50)
    # Sorted Dists: ~0.1, ~5.51, ~5.65, ~5.65, ~7.07, ~7.07
    # k=3 smallest: 0.1, 5.51, 5.65
    # Avg = (sqrt(0.01) + sqrt(30.41) + sqrt(32)) / 3 
    expected_novelty2 = (math.sqrt(0.01) + math.sqrt(3.9**2 + 3.9**2) + math.sqrt(4.0**2+4.0**2)) / 3 # Recalculated based on BVs
    novelty2 = calculate_novelty_score(pop_bvs[2], pop_bvs, archive, k)
    assert novelty2 == pytest.approx(expected_novelty2)

    # --- Test individual 4 (None) --- 
    novelty4 = calculate_novelty_score(pop_bvs[4], pop_bvs, archive, k)
    assert novelty4 == 0.0
    
    # --- Test no neighbors --- 
    empty_archive = NoveltyArchive()
    empty_pop = [None, [1.0, 1.0]]
    novelty_inf = calculate_novelty_score(empty_pop[1], empty_pop, empty_archive, k)
    assert novelty_inf == float('inf')

# --- Test Speciation ---

def test_assign_species():
    pop_spec_dna = [
        DNASequence("AAAAAAAA"), # 0 -> Species 0 Rep
        DNASequence("AAAAAAAT"), # 1 -> Species 0 (Dist 1)
        DNASequence("AAAATTTT"), # 2 -> Species 1 Rep (Dist 4 from 0)
        DNASequence("TTTTTTTT"), # 3 -> Species 2 Rep (Dist 8 from 0, 4 from 2)
        DNASequence("TTTTTTTA"), # 4 -> Species 2 (Dist 1 from 3)
        DNASequence("AATTCCGG"), # 5 -> Species 3 Rep (Dist 4 from 0, != S1 or S2)
        DNASequence("AATTCCGT")  # 6 -> Species 3 (Dist 1 from 5)
    ]
    spec_threshold = 2.0
    species_map = assign_species(pop_spec_dna, spec_threshold)
    
    assert len(species_map) == 4 # Expect 4 species
    assert species_map.get(0) == [0, 1]
    assert species_map.get(1) == [2]
    assert species_map.get(2) == [3, 4]
    assert species_map.get(3) == [5, 6]

def test_assign_species_all_same():
    pop_dna = [DNASequence("CCCC")] * 5
    species_map = assign_species(pop_dna, distance_threshold=1.0)
    assert len(species_map) == 1
    assert species_map.get(0) == [0, 1, 2, 3, 4]
    
def test_assign_species_all_different():
    pop_dna = [
        DNASequence("AAAA"),
        DNASequence("CCCC"),
        DNASequence("GGGG"),
        DNASequence("TTTT"),
    ]
    species_map = assign_species(pop_dna, distance_threshold=3.0) # Threshold < min distance (4)
    assert len(species_map) == 4
    assert species_map.get(0) == [0]
    assert species_map.get(1) == [1]
    assert species_map.get(2) == [2]
    assert species_map.get(3) == [3]

def test_assign_species_empty():
    assert assign_species([], 2.0) == {} 