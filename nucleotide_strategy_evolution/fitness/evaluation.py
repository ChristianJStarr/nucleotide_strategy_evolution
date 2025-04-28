"""Fitness evaluation functions and classes."""

from typing import Dict, Any, List, Tuple, Sequence, Optional
import random # For simulating results in Phase 1 & 2

from nucleotide_strategy_evolution.core.structures import Chromosome
from nucleotide_strategy_evolution.backtesting.interface import (
    AbstractBacktestEngine,
    BacktestingResults
)

# Type hint for fitness value
# For MOO, fitness is typically a tuple or list of objective scores
FitnessType = Tuple[float, ...]

class BasicFitnessEvaluator:
    """Calculates fitness based on backtest results (Phase 1: Single Objective)."""
    
    def __init__(self, backtester: AbstractBacktestEngine):
        """Initializes the evaluator with a backtesting engine."""
        if not isinstance(backtester, AbstractBacktestEngine):
            raise TypeError("backtester must be an instance of AbstractBacktestEngine")
        self.backtester = backtester

    def evaluate(self, chromosome: Chromosome, compliance_rules: Dict[str, Any]) -> float:
        """Runs backtest and returns a single fitness score (higher is better).

        Phase 1 Implementation Notes:
        - Uses the provided backtester (currently a placeholder).
        - Simulates extracting results (compliance, profit) as the backtester
          doesn't return real data yet.
        - Returns -inf if simulated compliance fails, otherwise simulated profit.

        Args:
            chromosome: The chromosome to evaluate.
            compliance_rules: Dictionary of rules passed to the backtester.

        Returns:
            A single float representing fitness (higher is better).
        """
        # 1. Run backtest using the provided engine
        # This currently uses the placeholder engine from backtesting.interface
        backtest_results: BacktestingResults = self.backtester.run(chromosome, compliance_rules)

        # --- Simulation for Phase 1 --- 
        # In later phases, these values would come from the actual backtest_results object
        
        # 2. Simulate Compliance Check (e.g., daily loss hit)
        # Randomly decide if a hard rule was violated for demonstration
        simulated_violation = random.random() < 0.1 # 10% chance of violation
        # setattr(backtest_results, 'simulated_hard_rule_violation', simulated_violation) # Can add to results if needed
        
        if simulated_violation or backtest_results.hard_rule_violation:
            # print(f"  Individual failed compliance check (simulated). Assigning worst fitness.")
            return -float('inf') # Assign poorest possible fitness

        # 3. Simulate extracting a single fitness metric (e.g., net profit)
        simulated_profit = random.uniform(-1000, 5000) # Simulate some profit/loss range
        # setattr(backtest_results, 'simulated_net_profit', simulated_profit) # Can add to results if needed
        # --- End Simulation --- 

        # Return the single objective fitness score
        return float(simulated_profit)

# --- Multi-Objective Fitness Evaluation (Phase 2+) ---

class MultiObjectiveEvaluator:
    """Calculates multiple fitness objectives based on backtest results."""

    def __init__(self, backtester: AbstractBacktestEngine, objectives: List[str]):
        """Initializes the evaluator.

        Args:
            backtester: An instance of AbstractBacktestEngine.
            objectives: A list of objective names (keys) to be calculated.
                        Signs indicate direction (e.g., 'net_profit', '-max_drawdown').
        """
        if not isinstance(backtester, AbstractBacktestEngine):
            raise TypeError("backtester must be an instance of AbstractBacktestEngine")
        if not objectives:
            raise ValueError("At least one objective must be specified.")
            
        self.backtester = backtester
        self.objectives = objectives
        # Store parsed objectives for easier access (objective name, minimization flag)
        self.parsed_objectives: List[Tuple[str, bool]] = []
        for obj_str in objectives:
            minimize = obj_str.startswith('-')
            name = obj_str[1:] if minimize else obj_str
            self.parsed_objectives.append((name, minimize))

    def evaluate(self, chromosome: Chromosome, compliance_rules: Dict[str, Any]) -> FitnessType:
        """Runs backtest and returns a tuple of objective scores.
        
        Phase 2 Implementation Notes:
        - Uses the provided backtester (placeholder).
        - Simulates extracting multiple results (compliance, profit, drawdown, etc.).
        - Returns a tuple of (-inf, +inf, ...) if compliance fails (adjusting signs based on objective).
        - Otherwise returns simulated objective values, adjusting signs for minimization.

        Args:
            chromosome: The chromosome to evaluate.
            compliance_rules: Dictionary of rules passed to the backtester.

        Returns:
            A tuple of floats representing the objective scores.
        """
        # 1. Run backtest
        backtest_results: BacktestingResults = self.backtester.run(chromosome, compliance_rules)
        
        # 2. Check Compliance from Backtest Results
        if backtest_results.hard_rule_violation:
            # Return worst possible values for each objective, considering minimize/maximize
            worst_fitness = []
            for _, minimize in self.parsed_objectives:
                worst_fitness.append(float('inf') if minimize else -float('inf'))
            # print(f"  Individual failed compliance check (from backtest). Assigning worst fitness: {tuple(worst_fitness)}")
            return tuple(worst_fitness)

        # --- Simulation of Objective Stats (Still needed until backtester returns real stats) ---
        # 3. Simulate extracting multiple objective values
        # These would normally come from backtest_results.stats or similar
        simulated_stats = {
            'net_profit': random.uniform(-1000, 5000),
            'max_drawdown': random.uniform(0.01, 0.5), # Typically positive
            'sortino_ratio': random.uniform(-1.0, 3.0),
            # Add other potential metrics here
        }
        # --- End Simulation --- 
        
        # 4. Build fitness tuple based on requested objectives
        fitness_values = []
        for name, minimize in self.parsed_objectives:
            if name not in simulated_stats:
                print(f"Warning: Objective '{name}' not found in simulated stats. Using 0.")
                value = 0.0
            else:
                value = simulated_stats[name]
            
            # Adjust sign for minimization objectives
            fitness_values.append(-value if minimize else value)
            
        return tuple(fitness_values)

    def evaluate_population(
        self,
        population: Dict[int, Chromosome], # Use dict for {index: chromosome}
        compliance_rules: Dict[str, Any],
        n_jobs: int = -1 # Number of parallel jobs (-1 uses all available cores)
    ) -> Dict[int, FitnessType]:
        """Evaluates the fitness of an entire population in parallel.

        Args:
            population: Dictionary mapping individual index to Chromosome object.
            compliance_rules: Dictionary of rules passed to the backtester.
            n_jobs: The number of CPU cores to use for parallel evaluation.
                    -1 means using all processors.
                    1 means no parallelism (useful for debugging).

        Returns:
            A dictionary mapping individual index to its calculated fitness tuple.
        """
        if n_jobs == 0:
             raise ValueError("n_jobs cannot be 0.")
             
        # Prepare arguments for parallel execution
        # We need to pass the chromosome and compliance rules to each worker process
        # The self.evaluate method needs to be called, but instance methods aren't directly
        # picklable in the same way simple functions are. We can use a helper function
        # or ensure the backtester itself is picklable and pass relevant parts.
        
        # Let's assume self.evaluate is picklable or can be called via a helper
        # Prepare tasks: List of tuples (chromosome, compliance_rules)
        tasks = [(chromo, compliance_rules) for chromo in population.values()]
        indices = list(population.keys())
        
        results: List[FitnessType] = []
        
        if n_jobs == 1:
             print("Evaluating population sequentially...")
             for i, (chromo, rules) in enumerate(tasks):
                 print(f" Evaluating individual {indices[i]}...")
                 results.append(self.evaluate(chromo, rules))
        else:
             print(f"Evaluating population in parallel using {n_jobs if n_jobs > 0 else 'all'} cores...")
             try:
                 # Import multiprocessing inside the method if needed, or at top level
                 import multiprocessing
                 from functools import partial
                 
                 pool_size = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
                 pool_size = min(pool_size, len(tasks)) # Don't use more cores than tasks
                 
                 # Use a partial function to wrap self.evaluate with fixed compliance_rules
                 # This helps with pickling if self.evaluate itself is complex
                 # Note: self.backtester must be picklable or handled carefully
                 eval_func = partial(self.evaluate, compliance_rules=compliance_rules)
                 
                 # Extract just the chromosomes for mapping
                 chromosomes_only = [chromo for chromo, _ in tasks]
                 
                 with multiprocessing.Pool(processes=pool_size) as pool:
                     # Map the evaluation function over the chromosomes
                     results = pool.map(eval_func, chromosomes_only)
                     
             except (ImportError, PicklingError) as e:
                 print(f"Warning: Parallel evaluation failed ({e}). Falling back to sequential evaluation.")
                 results = [self.evaluate(chromo, rules) for chromo, rules in tasks]
             except Exception as e:
                 print(f"An unexpected error occurred during parallel evaluation: {e}. Falling back to sequential.")
                 results = [self.evaluate(chromo, rules) for chromo, rules in tasks]

        # Combine results with original indices
        fitness_dict = {idx: fit for idx, fit in zip(indices, results)}
        return fitness_dict

# --- Example Usage ---
if __name__ == '__main__':
    from nucleotide_strategy_evolution.backtesting.interface import setup_backtester
    from nucleotide_strategy_evolution.encoding import random_dna_sequence, decode_chromosome

    # Setup dependencies
    backtester = setup_backtester("dummy_data_path")
    mock_rules = {"daily_loss": 1000}
    objectives_config = ['net_profit', '-max_drawdown', 'sortino_ratio']
    moo_evaluator = MultiObjectiveEvaluator(backtester, objectives=objectives_config)

    print("Testing MultiObjectiveEvaluator:")
    for i in range(5):
        dna = random_dna_sequence(100)
        chromo = decode_chromosome(dna)
        print(f"\nEvaluating individual {i} (DNA: {str(chromo.raw_dna)[:30]}...)")
        
        # Evaluate fitness
        fitness_tuple = moo_evaluator.evaluate(chromo, mock_rules)
        print(f"  Objectives ({objectives_config}): {fitness_tuple}")

    # Test evaluate_population
    print("\nTesting MultiObjectiveEvaluator.evaluate_population:")
    pop_size = 10
    mock_population = {i: decode_chromosome(random_dna_sequence(100)) for i in range(pop_size)}
    
    print("\nSequential Evaluation (n_jobs=1):")
    fitness_results_seq = moo_evaluator.evaluate_population(mock_population, mock_rules, n_jobs=1)
    assert len(fitness_results_seq) == pop_size
    print("  Sample results:", dict(list(fitness_results_seq.items())[:3]))

    print("\nParallel Evaluation (n_jobs=-1):")
    try:
        fitness_results_par = moo_evaluator.evaluate_population(mock_population, mock_rules, n_jobs=-1)
        assert len(fitness_results_par) == pop_size
        print("  Sample results:", dict(list(fitness_results_par.items())[:3]))
        # Note: Results might differ slightly due to random simulation in evaluate
        # A more robust test would use a deterministic backtester
    except Exception as e:
        print(f"Parallel evaluation test failed: {e}") 