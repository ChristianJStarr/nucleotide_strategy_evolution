"""Functions for measuring population diversity."""

from typing import List, Sequence, Dict, Optional
import random
import itertools
import math
import heapq
import pytest
import numpy as np

from nucleotide_strategy_evolution.core.structures import DNASequence, Chromosome, Gene
from nucleotide_strategy_evolution.fitness.evaluation import FitnessType
# from .population import Population # Avoid circular import if called from Population

# --- Genotypic Diversity Metrics ---

def calculate_hamming_distance(seq1: DNASequence, seq2: DNASequence) -> int:
    """Calculates the Hamming distance between two DNA sequences.
    
    The Hamming distance is the number of positions at which the corresponding
    symbols are different.
    Raises ValueError if sequences have different lengths.
    """
    if len(seq1) != len(seq2):
        # TODO: Decide how to handle variable length - pad, align, or use edit distance?
        # For now, require equal length.
        raise ValueError("Hamming distance requires sequences of equal length.")
        
    distance = 0
    for i in range(len(seq1)):
        if seq1.sequence[i] != seq2.sequence[i]:
            distance += 1
    return distance

def calculate_average_hamming_distance(dna_sequences: Sequence[DNASequence], sample_size: int = 100) -> float:
    """Calculates the average Hamming distance across a sample of pairs in the population.

    Args:
        dna_sequences: A list or sequence of DNASequence objects.
        sample_size: The number of pairs to sample for calculation.
                     Set to 0 or negative to calculate for all pairs (can be slow).

    Returns:
        The average Hamming distance, or 0.0 if fewer than 2 sequences.
    """
    n = len(dna_sequences)
    if n < 2:
        return 0.0

    total_distance = 0
    pairs_compared = 0

    if sample_size <= 0 or n * (n - 1) // 2 <= sample_size:
        # Calculate for all unique pairs
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    total_distance += calculate_hamming_distance(dna_sequences[i], dna_sequences[j])
                    pairs_compared += 1
                except ValueError:
                    # Handle sequences of different lengths if necessary (skipped for now)
                    print(f"Warning: Skipping Hamming distance for pair ({i}, {j}) due to unequal lengths.")
                    pass 
    else:
        # Sample pairs
        indices = list(range(n))
        for _ in range(sample_size):
            idx1, idx2 = random.sample(indices, 2)
            try:
                total_distance += calculate_hamming_distance(dna_sequences[idx1], dna_sequences[idx2])
                pairs_compared += 1
            except ValueError:
                 print(f"Warning: Skipping Hamming distance for pair ({idx1}, {idx2}) due to unequal lengths.")
                 pass

    if pairs_compared == 0:
        return 0.0 # Avoid division by zero (e.g., if all pairs had unequal lengths)
        
    return total_distance / pairs_compared

# --- Fitness Sharing (Genotypic) ---

def sharing_function(distance: int, sigma_share: float, alpha: float = 1.0) -> float:
    """Calculates the sharing value based on distance.
    
    Commonly uses a triangular sharing function: sh(d) = 1 - (d / sigma_share)^alpha if d < sigma_share, else 0.
    
    Args:
        distance: The distance (e.g., Hamming) between two individuals.
        sigma_share: The sharing radius. Individuals within this distance share fitness.
        alpha: Controls the shape of the sharing function (usually 1).
        
    Returns:
        The sharing value (0 to 1).
    """
    if sigma_share <= 0:
        return 0.0 # Avoid division by zero and invalid radius
    if distance < sigma_share:
        return max(0.0, 1.0 - (distance / sigma_share)**alpha)
    else:
        return 0.0

def apply_fitness_sharing(
    dna_sequences: Sequence[DNASequence], 
    original_fitness: Dict[int, FitnessType],
    sigma_share: float,
    alpha: float = 1.0,
    is_multi_objective: bool = False
) -> Dict[int, FitnessType]:
    """Adjusts fitness scores using genotypic fitness sharing.

    Divides each individual's fitness by its niche count, where the niche count
    is the sum of sharing function values with all other individuals.

    Args:
        dna_sequences: Sequence of DNASequence objects for the population.
        original_fitness: Dictionary mapping index to the original fitness score(s).
        sigma_share: The sharing radius for Hamming distance.
        alpha: Sharing function shape parameter.
        is_multi_objective: Flag indicating if fitness is a tuple (currently sharing 
                            is applied uniformly, might need refinement for MOO).

    Returns:
        A dictionary mapping index to the adjusted fitness score(s).
    """
    n = len(dna_sequences)
    if n == 0 or not original_fitness:
        return {}
        
    adjusted_fitness = {} # original_fitness.copy() # Start with original values
    niche_counts = {idx: 0.0 for idx in range(n)}
    
    # Calculate niche counts
    for i in range(n):
        # Check if individual i has fitness before calculating its niche
        if original_fitness.get(i) is None:
             continue 
             
        niche_counts[i] = 1.0 # Count self
        for j in range(n): # Compare with all others (including self, adds 1.0)
            if i == j: continue
            if original_fitness.get(j) is None: continue # Skip comparing with invalid individuals
            
            try:
                distance = calculate_hamming_distance(dna_sequences[i], dna_sequences[j])
                niche_counts[i] += sharing_function(distance, sigma_share, alpha)
            except ValueError:
                # Handle unequal lengths - they don't share based on this metric
                pass 
                
    # Adjust fitness scores
    for i in range(n):
        original_fit = original_fitness.get(i)
        if original_fit is None:
            adjusted_fitness[i] = None # Preserve None fitness
            continue
            
        niche_count = niche_counts.get(i, 1.0) # Get niche count, default to 1 if somehow missed
        if niche_count <= 0: # Avoid division by zero
            niche_count = 1.0 
            
        if is_multi_objective and isinstance(original_fit, Sequence):
            # Apply sharing uniformly to all objectives for now
            # More advanced MOO sharing might exist
            adj_fit_tuple = tuple(obj / niche_count for obj in original_fit)
            adjusted_fitness[i] = adj_fit_tuple
        elif isinstance(original_fit, (int, float)): # Single objective
            adjusted_fitness[i] = original_fit / niche_count
        else:
             print(f"Warning: Unexpected fitness type for individual {i}. Cannot apply sharing.")
             adjusted_fitness[i] = original_fit # Keep original if type unknown
             
    return adjusted_fitness

# --- Phenotypic/Behavioral Diversity (Placeholders) ---

# Import BehaviorVector type hint
from .behavior import BehaviorVector, characterize_behavior # Also import characterize
from ..backtesting.interface import BacktestingResults # Need for characterization call

def calculate_behavioral_distance(bv1: BehaviorVector, bv2: BehaviorVector, normalize: bool = False) -> float:
    """Calculates the Euclidean distance between two behavior vectors.

    Args:
        bv1: The first behavior vector (list of floats).
        bv2: The second behavior vector (list of floats).
        normalize: If True, normalize vectors before calculating distance (TODO).

    Returns:
        The Euclidean distance, or infinity if vectors have different lengths.
    """
    if len(bv1) != len(bv2):
         print(f"Warning: Behavior vectors have different lengths ({len(bv1)} vs {len(bv2)}). Cannot calculate distance.")
         return float('inf') # Return infinity for incompatible vectors
         
    if normalize:
         # TODO: Implement normalization (e.g., min-max scaling based on population stats)
         print("Warning: Behavioral vector normalization not yet implemented.")
         pass
         
    # Calculate squared Euclidean distance
    squared_diff_sum = 0.0
    for i in range(len(bv1)):
        diff = bv1[i] - bv2[i]
        squared_diff_sum += diff * diff
        
    return math.sqrt(squared_diff_sum)

# --- Novelty Search Components (Phase 3+) ---

class NoveltyArchive:
    """Stores behavioral characterizations of novel individuals found so far."""
    def __init__(self, capacity: int = 100):
        self.archive: List[BehaviorVector] = []
        self.capacity = capacity

    def add_to_archive(self, behavior_vector: BehaviorVector):
        """Adds a behavior vector to the archive, potentially managing capacity."""
        if not behavior_vector: return
        
        # Simple capacity management: remove oldest if full
        if len(self.archive) >= self.capacity:
            self.archive.pop(0) # Remove the oldest entry
        self.archive.append(behavior_vector)
        
    def get_archive_behaviors(self) -> List[BehaviorVector]:
        return self.archive
        
    def __len__(self) -> int:
         return len(self.archive)

def calculate_novelty_score(
    individual_bv: BehaviorVector,
    population_bvs: List[Optional[BehaviorVector]],
    archive: NoveltyArchive,
    k_neighbors: int = 10
) -> float:
    """Calculates the novelty score for an individual.
    
    Based on the average distance to its k-nearest neighbors in the current
    population and the novelty archive.
    
    Args:
        individual_bv: The behavior vector of the individual to score.
        population_bvs: List of behavior vectors for the current population 
                        (can include Nones or the individual itself).
        archive: The NoveltyArchive instance.
        k_neighbors: The number of nearest neighbors to consider.
        
    Returns:
        The novelty score (average distance to k-NN). Higher is more novel.
        Returns 0.0 if the individual's behavior vector is None.
    """
    if individual_bv is None:
         return 0.0
         
    # Combine population and archive behaviors for neighbor search
    # Exclude None values and the individual itself from the population list
    neighbors_bvs = archive.get_archive_behaviors() + \
                    [bv for bv in population_bvs if bv is not None and bv is not individual_bv]
                    
    if not neighbors_bvs:
        return float('inf') # Maximally novel if nothing to compare against
        
    # Find distances to all potential neighbors
    distances = []
    for neighbor_bv in neighbors_bvs:
        try:
             dist = calculate_behavioral_distance(individual_bv, neighbor_bv)
             if dist != float('inf'): # Ignore incompatible vectors
                  distances.append(dist)
        except Exception as e: # Catch potential errors in distance calc
             print(f"Warning: Error calculating distance for novelty: {e}")
             continue
             
    if not distances:
         return float('inf') # No comparable neighbors found
         
    # Find k nearest neighbors
    # Use heapq for efficiency if k is much smaller than N
    actual_k = min(k_neighbors, len(distances))
    if actual_k == 0:
        return float('inf') # Should not happen if distances list is not empty
        
    k_nearest_distances = heapq.nsmallest(actual_k, distances)
    
    # Calculate average distance to k-NN
    novelty_score = sum(k_nearest_distances) / actual_k
    
    return novelty_score

# --- Speciation (Genotypic) ---

def assign_species(
    dna_sequences: Sequence[DNASequence],
    distance_threshold: float, # e.g., sigma_share from fitness sharing
    min_species_size: int = 1
) -> Dict[int, List[int]]:
    """Assigns individuals to species based on genotypic distance.

    Uses a simple greedy clustering approach. The first individual forms species 1.
    Subsequent individuals join the first species whose representative they are
    close enough to, otherwise they form a new species.

    Args:
        dna_sequences: Sequence of DNASequence objects for the population.
        distance_threshold: Maximum Hamming distance to join an existing species.
        min_species_size: Minimum number of individuals required to form a valid species
                          (currently not strictly enforced in this simple version).

    Returns:
        A dictionary mapping species ID (starting from 0) to a list of
        individual indices belonging to that species.
    """
    n = len(dna_sequences)
    if n == 0:
        return {}

    species: Dict[int, List[int]] = {}
    representatives: Dict[int, DNASequence] = {} # Map species ID to representative DNA
    next_species_id = 0
    assigned = [False] * n

    for i in range(n):
        if assigned[i]:
            continue

        found_species = False
        # Try to assign to an existing species
        for species_id, rep_dna in representatives.items():
            try:
                distance = calculate_hamming_distance(dna_sequences[i], rep_dna)
                if distance <= distance_threshold:
                    species[species_id].append(i)
                    assigned[i] = True
                    found_species = True
                    break # Assigned to the first matching species
            except ValueError:
                # Handle unequal lengths - cannot belong to this species
                pass

        # If not assigned to any existing species, form a new one
        if not found_species:
            new_id = next_species_id
            species[new_id] = [i]
            representatives[new_id] = dna_sequences[i]
            assigned[i] = True
            next_species_id += 1
            
    # Optional: Filter out species smaller than min_species_size? 
    # This might leave some individuals unassigned if not handled carefully.
    # filtered_species = {sid: members for sid, members in species.items() if len(members) >= min_species_size}
    # return filtered_species

    return species

# --- Example Usage ---
if __name__ == '__main__':
    seq_a = DNASequence("ATGCATGC")
    seq_b = DNASequence("ATGGATAC") # 2 diffs
    seq_c = DNASequence("ATGCATGC") # 0 diffs
    seq_d = DNASequence("TGCATGCA") # 8 diffs
    seq_e = DNASequence("AAAAAAAA")
    seq_f = DNASequence("TTTTTTTT")
    seq_g = DNASequence("ATGC") # Different length

    print("--- Testing Hamming Distance ---")
    print(f"Dist(A, B): {calculate_hamming_distance(seq_a, seq_b)}") # Expected: 2
    print(f"Dist(A, C): {calculate_hamming_distance(seq_a, seq_c)}") # Expected: 0
    print(f"Dist(A, D): {calculate_hamming_distance(seq_a, seq_d)}") # Expected: 8
    print(f"Dist(E, F): {calculate_hamming_distance(seq_e, seq_f)}") # Expected: 8
    try:
        calculate_hamming_distance(seq_a, seq_g)
    except ValueError as e:
        print(f"Dist(A, G): Error as expected -> {e}")
        
    print("\n--- Testing Average Hamming Distance ---")
    pop_dna = [seq_a, seq_b, seq_c, seq_d, seq_e, seq_f]
    avg_dist_all = calculate_average_hamming_distance(pop_dna, sample_size=0)
    print(f"Avg Dist (All Pairs): {avg_dist_all:.2f}")
    
    avg_dist_sample = calculate_average_hamming_distance(pop_dna, sample_size=5)
    print(f"Avg Dist (Sample=5): {avg_dist_sample:.2f}")
    
    pop_dna_varied_len = [seq_a, seq_b, seq_g] # Includes different length
    avg_dist_varied = calculate_average_hamming_distance(pop_dna_varied_len, sample_size=0)
    print(f"Avg Dist (Varied Length, All Pairs): {avg_dist_varied:.2f}") # Should skip A<->G, B<->G 

    print("\n--- Testing Fitness Sharing ---")
    pop_sharing_dna = [
        DNASequence("AAAAAAAA"), # 0
        DNASequence("AAAAAAAT"), # 1 (Close to 0)
        DNASequence("AAAATTTT"), # 2 (Mid)
        DNASequence("TTTTTTTT"), # 3 (Far from 0, close to 4)
        DNASequence("TTTTTTTA")  # 4 (Close to 3)
    ]
    # Assume higher fitness is better
    original_fits = {
        0: 100.0,
        1: 95.0,
        2: 80.0,
        3: 70.0,
        4: 68.0
    }
    sigma = 3.0 # Share if distance < 3
    alpha_shape = 1.0
    
    print(f"Original Fitness: {original_fits}")
    print(f"Sigma Share: {sigma}")
    
    adj_fits = apply_fitness_sharing(pop_sharing_dna, original_fits, sigma, alpha_shape)
    print(f"Adjusted Fitness: {adj_fits}")
    # Expected: Fitness of 0 and 1 reduced significantly due to proximity.
    # Fitness of 3 and 4 reduced significantly due to proximity.
    # Fitness of 2 reduced less as it's further from the clusters.

    # Test MOO sharing
    original_fits_moo = {
        0: (100.0, 5.0),
        1: (95.0, 4.8),
        2: (80.0, 6.0),
        3: (70.0, 2.0),
        4: (68.0, 1.9) 
    }
    print(f"\nOriginal MOO Fitness: {original_fits_moo}")
    adj_fits_moo = apply_fitness_sharing(pop_sharing_dna, original_fits_moo, sigma, alpha_shape, is_multi_objective=True)
    print(f"Adjusted MOO Fitness: {adj_fits_moo}")
    # Expect both objectives to be scaled down based on niche count 

    print(f"\n--- Testing Behavioral Distance ---")
    behav1 = [1.5, 10.0, 0.6, 2.0] # AvgHold, NumTrades, WinRate, ProfitFactor
    behav2 = [1.8, 12.0, 0.5, 2.2] 
    behav3 = [1.5, 10.0, 0.6, 2.0] # Identical to behav1
    behav4 = [10.0, 1.0, 0.1, 0.5] # Very different
    behav5 = [1.5, 10.0] # Different length
    
    dist12 = calculate_behavioral_distance(behav1, behav2)
    dist13 = calculate_behavioral_distance(behav1, behav3)
    dist14 = calculate_behavioral_distance(behav1, behav4)
    dist15 = calculate_behavioral_distance(behav1, behav5)
    
    print(f"Dist(B1, B2): {dist12:.4f}") # Should be > 0
    print(f"Dist(B1, B3): {dist13:.4f}") # Should be 0.0
    print(f"Dist(B1, B4): {dist14:.4f}") # Should be large
    print(f"Dist(B1, B5): {dist15:.4f}") # Should be inf
    
    assert dist12 > 0
    assert dist13 == 0.0
    assert dist14 > dist12
    assert dist15 == float('inf') 

    print(f"\n--- Testing Novelty Search Components ---")
    archive = NoveltyArchive(capacity=5)
    pop_behaviors = [
        [1.0, 1.0], # 0
        [1.1, 1.1], # 1
        [5.0, 5.0], # 2
        [5.1, 5.0], # 3
        None,       # 4 (No behavior)
        [1.0, 1.0]  # 5 (Duplicate of 0)
    ]
    archive.add_to_archive([0.0, 0.0])
    archive.add_to_archive([10.0, 10.0])
    
    k = 3
    # Test individual 0 (close to 1 and 5 in pop, close to archive[0])
    novelty0 = calculate_novelty_score(pop_behaviors[0], pop_behaviors, archive, k)
    # Neighbors: [1], [5], archive[0], [2], [3], archive[1]
    # Dists: d(0,1)~0.14, d(0,5)=0, d(0,arc0)=1.41, d(0,2)~5.6, d(0,3)~5.7, d(0,arc1)=12.7
    # Nearest 3: 0, 0.14, 1.41 -> Avg = (0+0.14+1.41)/3 = 1.55/3 ~ 0.51
    print(f"Novelty(Ind 0): {novelty0:.4f}")
    assert novelty0 == pytest.approx( (0 + math.sqrt(0.02) + math.sqrt(2)) / 3, abs=1e-4 )

    # Test individual 2 (far from others in pop except 3, far from archive)
    novelty2 = calculate_novelty_score(pop_behaviors[2], pop_behaviors, archive, k)
    # Neighbors: [0], [1], [3], [5], archive[0], archive[1]
    # Dists: d(2,0)~5.6, d(2,1)~5.5, d(2,3)~0.1, d(2,5)=5.6, d(2,arc0)~7.07, d(2,arc1)~7.07
    # Nearest 3: 0.1, 5.5, 5.6 -> Avg = (0.1+5.5+5.6)/3 = 11.2/3 ~ 3.73
    print(f"Novelty(Ind 2): {novelty2:.4f}")
    assert novelty2 == pytest.approx( (math.sqrt(0.01) + math.sqrt(0.1**2+3.9**2) + math.sqrt(4.0**2+4.0**2)) / 3, abs=1e-4)

    # Test individual 4 (None behavior)
    novelty4 = calculate_novelty_score(pop_behaviors[4], pop_behaviors, archive, k)
    print(f"Novelty(Ind 4): {novelty4:.4f}")
    assert novelty4 == 0.0 

    print(f"\n--- Testing Speciation ---")
    pop_spec_dna = [
        DNASequence("AAAAAAAA"), # 0 -> Species 0 Rep
        DNASequence("AAAAAAAT"), # 1 -> Species 0 (Dist 1)
        DNASequence("AAAATTTT"), # 2 -> Species 1 Rep (Dist 4 from 0)
        DNASequence("TTTTTTTT"), # 3 -> Species 2 Rep (Dist 8 from 0, 4 from 2)
        DNASequence("TTTTTTTA"), # 4 -> Species 2 (Dist 1 from 3)
        DNASequence("AATTCCGG"), # 5 -> Species 3 Rep (Dist 4 from 0)
        DNASequence("AATTCCGT")  # 6 -> Species 3 (Dist 1 from 5)
    ]
    spec_threshold = 2.0
    species_map = assign_species(pop_spec_dna, spec_threshold)
    print(f"Speciation (Threshold={spec_threshold}):")
    for sid, members in sorted(species_map.items()):
        print(f"  Species {sid}: Indices {members}, Rep={pop_spec_dna[members[0]].sequence}")
    # Expected: {0: [0, 1], 1: [2], 2: [3, 4], 3: [5, 6]}
    assert species_map.get(0) == [0, 1]
    assert species_map.get(1) == [2]
    assert species_map.get(2) == [3, 4]
    assert species_map.get(3) == [5, 6] 