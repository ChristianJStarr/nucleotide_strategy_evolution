"""Functions for saving and loading evolution state."""

import pickle
import gzip
from typing import Any, Dict, List, Tuple, Optional

from .core.structures import DNASequence, Chromosome
from .fitness.ranking import FitnessType
from .population.behavior import BehaviorVector

# Define a structure for the state we want to save
EvolutionState = Dict[str, Any]

def save_evolution_state(
    filepath: str, 
    population_dna: List[DNASequence],
    fitness_scores: Optional[Dict[int, FitnessType]] = None,
    behavior_vectors: Optional[Dict[int, BehaviorVector]] = None,
    generation: Optional[int] = None,
    extra_data: Optional[Dict[str, Any]] = None,
    use_compression: bool = True
):
    """Saves the essential state of the evolution process to a file.

    Args:
        filepath: The path to the file where the state will be saved.
        population_dna: List of DNASequence objects representing the population.
        fitness_scores: Dictionary mapping individual index to fitness.
        behavior_vectors: Dictionary mapping individual index to behavior vector.
        generation: The current generation number.
        extra_data: Any other custom data to include in the state dictionary.
        use_compression: Whether to compress the output file using gzip.
    """
    state: EvolutionState = {
        'population_dna': population_dna,
        'fitness_scores': fitness_scores,
        'behavior_vectors': behavior_vectors,
        'generation': generation,
        'extra_data': extra_data or {},
    }
    
    print(f"Saving evolution state to {filepath}...")
    open_func = gzip.open if use_compression else open
    mode = 'wb' # Always write in binary mode for pickle
    
    try:
        with open_func(filepath, mode) as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Evolution state saved successfully.")
    except Exception as e:
        print(f"Error saving evolution state to {filepath}: {e}")
        raise # Re-raise the exception after printing

def load_evolution_state(filepath: str, use_compression: bool = True) -> EvolutionState:
    """Loads the evolution state from a file.

    Args:
        filepath: The path to the file containing the saved state.
        use_compression: Whether the file was saved with gzip compression.

    Returns:
        The loaded evolution state dictionary.
    """
    print(f"Loading evolution state from {filepath}...")
    open_func = gzip.open if use_compression else open
    mode = 'rb' # Always read in binary mode for pickle
    
    try:
        with open_func(filepath, mode) as f:
            state = pickle.load(f)
        print("Evolution state loaded successfully.")
        # Basic validation (can be expanded)
        if not isinstance(state, dict) or 'population_dna' not in state:
             raise ValueError("Loaded file does not appear to be a valid evolution state.")
        return state
    except FileNotFoundError:
        print(f"Error: State file not found at {filepath}")
        raise
    except (pickle.UnpicklingError, EOFError, gzip.BadGzipFile) as e:
        print(f"Error loading or unpickling state file {filepath}: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading state from {filepath}: {e}")
        raise

# --- Example Usage ---
if __name__ == '__main__':
    import os
    
    # Create dummy data
    pop_dna = [DNASequence("ATGC"), DNASequence("CGTA"), DNASequence("AATT")]
    fits = {0: (10.0, -1.0), 1: (8.5, -2.5), 2: (11.0, -0.5)}
    bvs = {0: [0.5, 10], 1: [0.7, 8], 2: [0.4, 12]}
    gen = 50
    extra = {'rng_state': random.getstate()} # Example extra data
    
    save_dir = "./temp_state"
    os.makedirs(save_dir, exist_ok=True)
    save_file_compressed = os.path.join(save_dir, "evolution_state.pkl.gz")
    save_file_uncompressed = os.path.join(save_dir, "evolution_state.pkl")

    # Test saving compressed
    print("\n--- Testing Save (Compressed) ---")
    try:
        save_evolution_state(save_file_compressed, pop_dna, fits, bvs, gen, extra, use_compression=True)
        assert os.path.exists(save_file_compressed)
    except Exception as e:
        print(f"Save compressed failed: {e}")

    # Test loading compressed
    print("\n--- Testing Load (Compressed) ---")
    try:
        loaded_state_gz = load_evolution_state(save_file_compressed, use_compression=True)
        assert loaded_state_gz['generation'] == gen
        assert len(loaded_state_gz['population_dna']) == len(pop_dna)
        assert loaded_state_gz['population_dna'][0].sequence == pop_dna[0].sequence
        assert loaded_state_gz['fitness_scores'] == fits
        assert loaded_state_gz['behavior_vectors'] == bvs
        assert 'rng_state' in loaded_state_gz['extra_data']
        print("Loaded compressed state seems OK.")
    except Exception as e:
        print(f"Load compressed failed: {e}")

    # Test saving uncompressed
    print("\n--- Testing Save (Uncompressed) ---")
    try:
        save_evolution_state(save_file_uncompressed, pop_dna, fits, bvs, gen, extra, use_compression=False)
        assert os.path.exists(save_file_uncompressed)
    except Exception as e:
        print(f"Save uncompressed failed: {e}")

    # Test loading uncompressed
    print("\n--- Testing Load (Uncompressed) ---")
    try:
        loaded_state_raw = load_evolution_state(save_file_uncompressed, use_compression=False)
        assert loaded_state_raw['generation'] == gen
        assert loaded_state_raw['population_dna'][0].sequence == pop_dna[0].sequence
        print("Loaded uncompressed state seems OK.")
    except Exception as e:
        print(f"Load uncompressed failed: {e}")
        
    # Test loading non-existent file
    print("\n--- Testing Load (Non-existent) ---")
    try:
         load_evolution_state("non_existent_file.pkl.gz")
    except FileNotFoundError:
         print("Caught FileNotFoundError as expected.")
    except Exception as e:
         print(f"Caught unexpected error: {e}")
         
    # Clean up dummy files
    print("\nCleaning up temporary files...")
    if os.path.exists(save_file_compressed):
        os.remove(save_file_compressed)
    if os.path.exists(save_file_uncompressed):
        os.remove(save_file_uncompressed)
    if os.path.exists(save_dir):
        try:
             os.rmdir(save_dir) # Only removes if empty
        except OSError:
             pass # Directory might not be empty if tests failed
    print("Cleanup finished.") 