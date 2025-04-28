"""Implements the Island Model for parallel evolution."""

from typing import List, Dict, Any, Optional
import random

from .population import Population
from nucleotide_strategy_evolution.fitness.evaluation import (
    BasicFitnessEvaluator, MultiObjectiveEvaluator
)
from nucleotide_strategy_evolution.fitness.ranking import fast_non_dominated_sort # For 'best' migration
# Need Chromosome for type hint
from nucleotide_strategy_evolution.core.structures import Chromosome

class IslandModel:
    """Manages multiple populations (islands) evolving in parallel with migration."""
    
    def __init__(self, config: Dict[str, Any], global_config: Dict[str, Any]):
        """Initializes the islands based on configuration.
        
        Args:
            config: Island-specific configuration (num_islands, migration, etc.).
            global_config: Overall evolution parameters (pop_size per island, dna_length, etc.).
                           This assumes pop_size in global_config applies *per island*.
        """
        self.num_islands = config.get('num_islands', 4)
        self.migration_frequency = config.get('migration_frequency', 10)
        self.num_migrants = config.get('num_migrants', 2)
        self.migration_selection_method = config.get('migration_selection_method', 'best')
        
        if self.num_islands <= 0:
             raise ValueError("Number of islands must be positive.")
        if self.migration_frequency <= 0:
             raise ValueError("Migration frequency must be positive.")
        if self.num_migrants < 0:
            raise ValueError("Number of migrants cannot be negative.")
            
        # Extract relevant global parameters needed for population init
        pop_config = global_config.get('population', {})
        self.pop_size_per_island = pop_config.get('size', 100)
        self.dna_length = pop_config.get('dna_length', 300)
        
        if self.num_migrants >= self.pop_size_per_island:
            print(f"Warning: num_migrants ({self.num_migrants}) >= pop_size_per_island ({self.pop_size_per_island}). Adjusting num_migrants.")
            self.num_migrants = max(0, self.pop_size_per_island - 1)

        print(f"Initializing Island Model with {self.num_islands} islands.")
        self.islands: List[Population] = []
        for i in range(self.num_islands):
            print(f"  Initializing Island {i} (Size: {self.pop_size_per_island})...")
            island_pop = Population(size=self.pop_size_per_island, dna_length=self.dna_length)
            island_pop.initialize() # Initialize with random individuals
            self.islands.append(island_pop)
            
    def get_island(self, index: int) -> Population:
        return self.islands[index]
        
    def __len__(self) -> int:
         return self.num_islands
         
    def evaluate_all(self, evaluator: Any, compliance_rules: Dict[str, Any]):
        """Evaluates the fitness of all individuals across all islands."""
        print(f"\nEvaluating fitness across {self.num_islands} islands...")
        # TODO: Parallelize this across islands (e.g., using multiprocessing)
        for i, island_pop in enumerate(self.islands):
            print(f"-- Evaluating Island {i} --")
            island_pop.evaluate_fitnesses(evaluator, compliance_rules)
            
    def migrate(self):
        """Performs migration between islands based on the configured strategy."""
        print(f"\nPerforming migration (Num Migrants: {self.num_migrants}, Method: {self.migration_selection_method})...")
        if self.num_migrants == 0 or self.num_islands < 2:
            print("Migration skipped (num_migrants=0 or < 2 islands).")
            return

        migrants_per_island: List[List[Chromosome]] = [[] for _ in range(self.num_islands)]

        # 1. Select migrants from each island
        for i, island_pop in enumerate(self.islands):
            candidates_indices = list(range(len(island_pop)))
            if not candidates_indices: continue # Skip empty islands
            
            selected_migrant_indices = []
            if self.migration_selection_method == 'best':
                # Select the best N individuals based on fitness
                # Requires fitness scores to be available and comparable
                valid_fitness = {idx: island_pop.get_fitness(idx) for idx in candidates_indices 
                                 if island_pop.get_fitness(idx) is not None}
                if not valid_fitness: continue # Skip if no valid fitness
                
                # Check if multi-objective
                is_moo = isinstance(next(iter(valid_fitness.values())), tuple) 
                
                if is_moo:
                    # Use NSGA-II ranking to find the best
                    fronts = fast_non_dominated_sort(valid_fitness)
                    best_indices_ranked = []
                    for front in fronts:
                         best_indices_ranked.extend(front)
                    selected_migrant_indices = best_indices_ranked[:self.num_migrants]
                else:
                    # Single objective: Sort by fitness (descending)
                    sorted_indices = sorted(valid_fitness.keys(), 
                                            key=lambda idx: valid_fitness[idx], 
                                            reverse=True)
                    selected_migrant_indices = sorted_indices[:self.num_migrants]
                    
            elif self.migration_selection_method == 'random':
                num_to_select = min(self.num_migrants, len(candidates_indices))
                selected_migrant_indices = random.sample(candidates_indices, num_to_select)
            else:
                print(f"Warning: Unknown migration selection method '{self.migration_selection_method}'. Defaulting to random.")
                num_to_select = min(self.num_migrants, len(candidates_indices))
                selected_migrant_indices = random.sample(candidates_indices, num_to_select)

            # Store the actual Chromosome objects to migrate
            migrants_per_island[i] = [island_pop[idx] for idx in selected_migrant_indices]
            # print(f"  Selected {len(migrants_per_island[i])} migrants from Island {i}") # Debug

        # 2. Perform migration (e.g., ring topology: Island i sends to i+1)
        if not any(migrants_per_island): # Check if any migrants were selected
            print("Migration skipped: No migrants selected (perhaps due to lack of valid fitness scores).")
            return
            
        for i in range(self.num_islands):
            source_island_idx = i
            target_island_idx = (i + 1) % self.num_islands # Ring topology
            
            migrants_to_send = migrants_per_island[source_island_idx]
            if not migrants_to_send:
                continue
                
            target_pop = self.islands[target_island_idx]
            
            # Replace the worst individuals in the target island
            num_to_replace = min(len(migrants_to_send), len(target_pop))
            if num_to_replace == 0: continue
            
            # Find worst individuals in target island
            target_valid_fitness = {idx: target_pop.get_fitness(idx) for idx in range(len(target_pop)) 
                                    if target_pop.get_fitness(idx) is not None}
            
            if not target_valid_fitness: # Cannot determine worst if no fitness
                # Replace random individuals if no fitness info
                indices_to_replace = random.sample(range(len(target_pop)), num_to_replace)
            else:
                 is_moo_target = isinstance(next(iter(target_valid_fitness.values())), tuple)
                 if is_moo_target:
                     # Rank and take from the worst front(s)
                     target_fronts = fast_non_dominated_sort(target_valid_fitness)
                     worst_indices_ranked = []
                     for front in reversed(target_fronts):
                         worst_indices_ranked.extend(front)
                     indices_to_replace = worst_indices_ranked[:num_to_replace]
                 else:
                      # Single objective: Sort ascending
                      sorted_target_indices = sorted(target_valid_fitness.keys(), 
                                                     key=lambda idx: target_valid_fitness[idx])
                      indices_to_replace = sorted_target_indices[:num_to_replace]
                      
            print(f"  Migrating {num_to_replace} individuals from Island {source_island_idx} to Island {target_island_idx} (replacing indices {indices_to_replace}).")
            for k in range(num_to_replace):
                replace_idx = indices_to_replace[k]
                # Replace individual and reset fitness score
                target_pop.individuals[replace_idx] = migrants_to_send[k]
                target_pop.set_fitness(replace_idx, None) 

# --- Example Usage ---
if __name__ == '__main__':
    # Example config (could load from YAML)
    island_conf = {
        'num_islands': 3,
        'migration_frequency': 5,
        'num_migrants': 1,
        'migration_selection_method': 'best'
    }
    global_conf = {
        'population': {'size': 10, 'dna_length': 40}
    }
    
    island_model = IslandModel(island_conf, global_conf)
    print(f"Created Island Model with {len(island_model)} islands.")
    
    # Simulate evaluation (need mock evaluator)
    class MockEvaluator: # Basic single-objective mock
         def evaluate(self, chromosome, rules):
             return random.random() * 100 
             
    mock_eval = MockEvaluator()
    mock_rules = {}
    island_model.evaluate_all(mock_eval, mock_rules)
    
    # Show fitness before migration
    print("\nFitness before migration:")
    for i, island in enumerate(island_model.islands):
        fits = [f"{island.get_fitness(j):.2f}" if island.get_fitness(j) is not None else "None" for j in range(len(island))]
        print(f" Island {i}: {fits}")
        
    # Perform migration
    island_model.migrate()
    
    print("\nFitness after migration (some should be None):")
    for i, island in enumerate(island_model.islands):
        fits = [f"{island.get_fitness(j):.2f}" if island.get_fitness(j) is not None else "None" for j in range(len(island))]
        print(f" Island {i}: {fits}") 