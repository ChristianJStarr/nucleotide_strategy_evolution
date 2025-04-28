"""Population management for the genetic algorithm."""

from typing import List, Optional, Dict, Any
import random

from nucleotide_strategy_evolution.core.structures import Chromosome, DNASequence
from nucleotide_strategy_evolution.encoding import random_dna_sequence, decode_chromosome
# Allow importing either evaluator type
from nucleotide_strategy_evolution.fitness.evaluation import BasicFitnessEvaluator, MultiObjectiveEvaluator, FitnessType

# Default length for randomly generated DNA sequences if not specified
DEFAULT_DNA_LENGTH = 300 

class Population:
    """Represents a population of individuals (chromosomes) for evolution."""

    def __init__(self, size: int, dna_length: int = DEFAULT_DNA_LENGTH):
        """Initializes an empty population.

        Args:
            size: The target number of individuals in the population.
            dna_length: The length of DNA sequences for randomly generated individuals.
        """
        if size <= 0:
            raise ValueError("Population size must be positive.")
        if dna_length <= 0:
             raise ValueError("DNA length must be positive.")
             
        self.target_size = size
        self.dna_length = dna_length
        # Store chromosomes directly. Fitness will be associated later.
        self.individuals: List[Chromosome] = [] 
        # Store fitness scores mapped to individuals (e.g., by id or index)
        self.fitness_scores: Dict[int, FitnessType] = {} # Use FitnessType hint
        # TODO: Consider mapping fitness scores to Chromosome objects directly or using IDs
        
    def initialize(self, strategy_template: Optional[Chromosome] = None):
        """Populates the individuals list with randomly generated chromosomes.
        
        Args:
            strategy_template: An optional template chromosome to seed the population.
                               (Currently unused in this basic implementation).
        """
        self.individuals = []
        self.fitness_scores = {} # Reset fitness scores on initialization
        print(f"Initializing population with {self.target_size} random individuals (DNA length: {self.dna_length}).")
        for i in range(self.target_size):
            # 1. Generate random DNA
            dna = random_dna_sequence(self.dna_length)
            # 2. Decode into a Chromosome (genes might be empty if no start/stop found)
            chromosome = decode_chromosome(dna)
            # TODO: Add seeding logic using strategy_template later
            self.individuals.append(chromosome)
            # Initialize fitness score placeholder (e.g., None or 0)
            # Using index as key for now, might need more robust ID later
            self.fitness_scores[i] = None 

    def __len__(self) -> int:
        """Returns the current number of individuals in the population."""
        return len(self.individuals)

    def __getitem__(self, index: int) -> Chromosome:
        """Gets the individual chromosome at the specified index."""
        return self.individuals[index]

    def get_fitness(self, index: int) -> Optional[FitnessType]:
        """Gets the fitness score (tuple for MOO) for the individual at the specified index."""
        return self.fitness_scores.get(index, None) 

    def set_fitness(self, index: int, fitness: FitnessType):
        """Sets the fitness score for the individual at the specified index."""
        if index >= len(self.individuals):
            raise IndexError("Index out of bounds for setting fitness.")
        self.fitness_scores[index] = fitness
        
    def evaluate_fitnesses(
        self, 
        fitness_evaluator: Any, # Accept either Basic or MultiObjective for now
        compliance_rules: Dict[str, Any]
    ):
        """Evaluates and stores fitness for all individuals using a fitness evaluator."""
        
        # Determine if it's multi-objective based on the evaluator type
        is_multi_objective = isinstance(fitness_evaluator, MultiObjectiveEvaluator)
        evaluator_type_name = type(fitness_evaluator).__name__
        
        # Basic type check
        if not isinstance(fitness_evaluator, (BasicFitnessEvaluator, MultiObjectiveEvaluator)):
             raise TypeError(f"Unsupported fitness evaluator type: {evaluator_type_name}")
            
        print(f"Evaluating fitness ({evaluator_type_name}) for {len(self.individuals)} individuals...")
        # TODO: Parallelize this evaluation later (Phase 4)
        for i, chromosome in enumerate(self.individuals):
            try:
                # Call the evaluator's evaluate method
                fitness_value = fitness_evaluator.evaluate(chromosome, compliance_rules)
                # print(f"  Individual {i}: Fitness = {fitness_value}") # Less verbose during eval
                self.set_fitness(i, fitness_value)
            except Exception as e:
                print(f"Error evaluating fitness for individual {i}: {e}")
                # Assign worst possible fitness based on evaluation type
                if is_multi_objective:
                    # Need number of objectives - get from evaluator
                    num_obj = len(fitness_evaluator.objectives)
                    worst_fitness = []
                    for _, minimize in fitness_evaluator.parsed_objectives:
                         worst_fitness.append(float('inf') if minimize else -float('inf'))
                    self.set_fitness(i, tuple(worst_fitness))
                else:
                    # Assign single worst fitness for BasicEvaluator
                    self.set_fitness(i, -float('inf')) 
                
        print("Fitness evaluation complete.")

# --- Example Usage Update ---
if __name__ == '__main__':
    from nucleotide_strategy_evolution.backtesting.interface import setup_backtester
    # Import both evaluators for the example
    from nucleotide_strategy_evolution.fitness.evaluation import BasicFitnessEvaluator, MultiObjectiveEvaluator

    pop_size = 10
    dna_len = 50
    population = Population(size=pop_size, dna_length=dna_len)
    population.initialize()
    print(f"Population initialized. Size: {len(population)}")
    
    # --- Test with Basic Evaluator --- 
    print("\nTesting evaluation with BasicFitnessEvaluator:")
    backtester_basic = setup_backtester("dummy_data_path")
    evaluator_basic = BasicFitnessEvaluator(backtester_basic)
    mock_rules = {"daily_loss": 100}
    population.evaluate_fitnesses(evaluator_basic, mock_rules) 
    print("Fitness scores (first 5):")
    for i in range(min(5, len(population))):
        print(f"  Individual {i}: Fitness = {population.get_fitness(i)}") 
        
    # --- Test with Multi-Objective Evaluator --- 
    print("\nTesting evaluation with MultiObjectiveEvaluator:")
    backtester_moo = setup_backtester("dummy_data_path_2") # Use same or different backtester
    objectives_config = ['net_profit', '-max_drawdown', 'sortino_ratio']
    evaluator_moo = MultiObjectiveEvaluator(backtester_moo, objectives=objectives_config)
    # Re-initialize fitness scores before MOO evaluation
    for i in range(len(population)):
        population.set_fitness(i, None) 
        
    population.evaluate_fitnesses(evaluator_moo, mock_rules)
    print(f"Fitness scores (first 5) for objectives {objectives_config}:")
    for i in range(min(5, len(population))):
        print(f"  Individual {i}: Fitness = {population.get_fitness(i)}") 