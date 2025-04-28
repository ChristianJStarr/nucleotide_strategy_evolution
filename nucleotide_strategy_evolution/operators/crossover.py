"""Crossover operators for DNA sequences."""

import random
import functools
from typing import Tuple, Dict, Any

from nucleotide_strategy_evolution.core.structures import DNASequence

# --- Crossover Operator Implementations ---

def single_point_crossover(parent1: DNASequence, parent2: DNASequence) -> Tuple[DNASequence, DNASequence]:
    """Performs single-point crossover on two DNA sequences.

    Chooses a random crossover point and swaps the segments after that point
    between the two parent sequences.

    Args:
        parent1: The first parent DNA sequence.
        parent2: The second parent DNA sequence.

    Returns:
        A tuple containing two offspring DNA sequences.
    """
    # Ensure sequences have the same length for simple crossover
    # TODO: Handle variable length sequences later (e.g., padding, alignment, or segment swapping)
    if len(parent1) != len(parent2):
        raise ValueError("Single-point crossover requires sequences of the same length.")
        
    n = len(parent1)
    if n < 2:
        # Cannot perform crossover on sequences shorter than 2
        return parent1, parent2 

    # Choose a random crossover point (index from 1 to n-1)
    crossover_point = random.randint(1, n - 1)

    # Create offspring
    offspring1_seq = parent1.sequence[:crossover_point] + parent2.sequence[crossover_point:]
    offspring2_seq = parent2.sequence[:crossover_point] + parent1.sequence[crossover_point:]

    return DNASequence(sequence=offspring1_seq), DNASequence(sequence=offspring2_seq)

def uniform_crossover(parent1: DNASequence, parent2: DNASequence, swap_prob: float = 0.5) -> Tuple[DNASequence, DNASequence]:
    """Performs uniform crossover on two DNA sequences.

    Iterates through the sequences and swaps nucleotides at each position
    with a probability `swap_prob`.

    Args:
        parent1: The first parent DNA sequence.
        parent2: The second parent DNA sequence.
        swap_prob: The probability (0.0 to 1.0) of swapping nucleotides at each position.

    Returns:
        A tuple containing two offspring DNA sequences.
    """
    if len(parent1) != len(parent2):
        raise ValueError("Uniform crossover requires sequences of the same length.")
    if not 0.0 <= swap_prob <= 1.0:
         raise ValueError("Swap probability must be between 0.0 and 1.0")
         
    n = len(parent1)
    offspring1_list = list(parent1.sequence)
    offspring2_list = list(parent2.sequence)

    for i in range(n):
        if random.random() < swap_prob:
            # Swap nucleotides
            offspring1_list[i], offspring2_list[i] = offspring2_list[i], offspring1_list[i]

    return DNASequence(sequence="".join(offspring1_list)), DNASequence(sequence="".join(offspring2_list))

# --- Operator Registry (Similar to Gene Factory) ---

CROSSOVER_REGISTRY: Dict[str, Any] = {
    "single_point": single_point_crossover,
    "uniform": uniform_crossover,
    # Add other crossover types here as implemented (e.g., multi_point, uniform)
}

def get_crossover_operator(config: Dict[str, Any]) -> Any:
    """Gets the crossover function based on configuration.

    Args:
        config: Dictionary typically containing {'type': 'operator_name', ...}

    Returns:
        The crossover function.
    """
    operator_type = config.get("type", "single_point") # Default to single_point
    if operator_type not in CROSSOVER_REGISTRY:
        raise ValueError(f"Unknown crossover operator type: {operator_type}")
    
    # TODO: Pass relevant config parameters to the operator function if needed
    # e.g., number of points for multi-point crossover, swap_prob for uniform
    operator_func = CROSSOVER_REGISTRY[operator_type]
    
    # Use partial application to bind configuration parameters
    kwargs = {}
    if operator_type == "uniform":
        kwargs['swap_prob'] = config.get("swap_prob", 0.5)
        
    if kwargs:
        return functools.partial(operator_func, **kwargs)
    else:
        return operator_func

# --- Example Usage ---
if __name__ == '__main__':
    p1 = DNASequence("AAAAAAAAAA")
    p2 = DNASequence("CCCCCCCCCC")
    print(f"Parent 1: {p1}")
    print(f"Parent 2: {p2}")
    
    o1, o2 = single_point_crossover(p1, p2)
    print("\nAfter Single-Point Crossover:")
    print(f"Offspring 1: {o1}")
    print(f"Offspring 2: {o2}")

    # Test getting operator from config
    config_sp = {"type": "single_point"}
    op_sp = get_crossover_operator(config_sp)
    o3, o4 = op_sp(p1, p2)
    print(f"\nUsing get_crossover_operator('{config_sp['type']}'): {o3}, {o4}")

    # Test Uniform Crossover
    p3 = DNASequence("AGAGAGAGAG")
    p4 = DNASequence("TCTCTCTCTC")
    print(f"\nParent 3: {p3}")
    print(f"Parent 4: {p4}")
    o5, o6 = uniform_crossover(p3, p4, swap_prob=0.5)
    print("\nAfter Uniform Crossover (prob=0.5):")
    print(f"Offspring 5: {o5}")
    print(f"Offspring 6: {o6}")
    
    config_uni = {"type": "uniform", "swap_prob": 0.2}
    op_uni = get_crossover_operator(config_uni)
    o7, o8 = op_uni(p3, p4)
    print(f"\nUsing get_crossover_operator('{config_uni}'): {o7}, {o8}")

    # Example of future config
    # config_mp = {"type": "multi_point", "points": 2}
    # op_mp = get_crossover_operator(config_mp) 