"""Example script demonstrating Walk-Forward Optimization (WFO)."""

import random
import os
import sys
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from nucleotide_strategy_evolution.population import Population
from nucleotide_strategy_evolution.population.island import IslandModel # Can use islands within WFO
from nucleotide_strategy_evolution.population.selection import get_selection_operator
from nucleotide_strategy_evolution.population.diversity import (
    calculate_average_hamming_distance, apply_fitness_sharing, 
    NoveltyArchive, calculate_novelty_score 
)
from nucleotide_strategy_evolution.population.behavior import characterize_behavior, BehaviorVector
from nucleotide_strategy_evolution.operators.crossover import get_crossover_operator
from nucleotide_strategy_evolution.operators.mutation import get_mutation_operator
from nucleotide_strategy_evolution.operators.adaptive import adapt_rates
from nucleotide_strategy_evolution.fitness.evaluation import MultiObjectiveEvaluator
from nucleotide_strategy_evolution.fitness.ranking import fast_non_dominated_sort, FitnessType
from nucleotide_strategy_evolution.backtesting.interface import setup_backtester, BacktestingResults
from nucleotide_strategy_evolution.validation import generate_wfo_splits
from nucleotide_strategy_evolution.utils.config_loader import load_config
from nucleotide_strategy_evolution.core.structures import Chromosome, DNASequence
from nucleotide_strategy_evolution.visualization.plotting import plot_pareto_front_2d

# --- Configuration --- 
EVO_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'evolution_params.yaml')
RULES_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'compliance_rules.yaml')
# TODO: Use real data source config
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'sample_data.csv') # Placeholder

# --- WFO Main Function --- 

def run_walk_forward_evolution():
    """Performs walk-forward optimization by running evolution on sequential training sets."""
    
    # 1. Load Configuration
    try:
        evo_config = load_config(EVO_CONFIG_PATH)
        compliance_rules = load_config(RULES_CONFIG_PATH)
        print("Configuration loaded.")
    except Exception as e:
        print(f"Error loading/parsing configuration: {e}")
        return
        
    # --- Extract Config Parameters --- 
    # Evolution Params
    pop_size = evo_config.get('population', {}).get('size', 50) # Smaller pop for faster WFO steps?
    dna_length = evo_config.get('population', {}).get('dna_length', 300)
    generations = evo_config.get('evolution', {}).get('generations', 20) # Fewer generations per WFO step
    crossover_config = evo_config.get('operators', {}).get('crossover', {'type': 'single_point', 'rate': 0.7})
    mutation_configs = evo_config.get('operators', {}).get('mutation', [{'type': 'point_mutation', 'rate': 0.1}])
    selection_config = evo_config.get('selection', {'method': 'nsga2'}) 
    objectives = evo_config.get('fitness', {}).get('objectives', ['net_profit', '-max_drawdown'])
    diversity_config = evo_config.get('diversity', {})
    sharing_config = diversity_config.get('fitness_sharing', {'enabled': False})
    adaptive_rate_config = evo_config.get('adaptive_rates', {'enabled': False})
    island_config = evo_config.get('island_model', {'enabled': False})
    # WFO Params
    wfo_config = evo_config.get('wfo', {})
    wfo_enabled = wfo_config.get('enabled', True) # Default WFO to true for this script
    train_periods = wfo_config.get('train_periods', 500) # e.g., 500 bars/days
    test_periods = wfo_config.get('test_periods', 100)
    step_periods = wfo_config.get('step_periods', test_periods) # Slide by OOS length
    anchored = wfo_config.get('anchored', False)
    # Backtesting Params
    backtest_engine_type = evo_config.get('backtesting', {}).get('engine', 'backtesting.py') 
    backtest_config = evo_config.get('backtesting', {}).get('config', {})
    
    if not wfo_enabled:
        print("WFO is disabled in config. Running basic evolution instead.")
        # Optionally call the basic_evolution script's function here
        # from basic_evolution import run_moo_evolution
        # run_moo_evolution()
        return
        
    # 2. Load Full Dataset (Placeholder)
    print(f"Loading full dataset from {DATA_PATH}...")
    # Replace with actual data loading
    full_dates = pd.date_range(start='2020-01-01', periods=train_periods + test_periods + step_periods * 5, freq='B') # Example length
    full_data = pd.DataFrame({
        'Open': np.random.rand(len(full_dates)) * 10 + 100,
        'High': lambda df: df['Open'] + np.random.rand(len(df)) * 2,
        'Low': lambda df: df['Open'] - np.random.rand(len(df)) * 2,
        'Close': lambda df: df['Open'] + np.random.randn(len(df)),
        'Volume': np.random.randint(100, 10000, size=len(full_dates))
    }, index=full_dates)
    # Ensure columns match backtesting.py requirements
    full_data['High'] = full_data[['Open', 'High']].max(axis=1)
    full_data['Low'] = full_data[['Open', 'Low']].min(axis=1)
    print(f"Full dataset loaded: {len(full_data)} rows from {full_data.index[0].date()} to {full_data.index[-1].date()}")

    # 3. Setup Shared Components (Operators)
    crossover_op = get_crossover_operator(crossover_config)
    mutation_ops_config = []
    for m_config in mutation_configs:
        try:
            op_func = get_mutation_operator(m_config)
            rate = m_config.get('rate')
            if rate is not None:
                 mutation_ops_config.append((op_func, m_config))
            else:
                 print(f"Warning: Mutation config missing 'rate' for type '{m_config.get('type')}'. Skipping.")
        except ValueError as e:
            print(f"Warning: Error getting mutation operator: {e}")
    parent_selector_fitness = get_selection_operator(selection_config)
    
    # --- WFO Loop --- 
    all_oos_results: List[BacktestingResults] = []
    all_best_chromosomes: List[Chromosome] = [] # Store best from each IS run
    wfo_split_generator = generate_wfo_splits(full_data, train_periods, test_periods, step_periods, anchored)
    
    for i, (train_data, test_data) in enumerate(wfo_split_generator):
        print(f"\n===== WFO Step {i+1}: Train {train_data.index[0].date()}-{train_data.index[-1].date()}, Test {test_data.index[0].date()}-{test_data.index[-1].date()} ====")
        
        # --- Setup for this WFO Step --- 
        # Re-initialize backtester with TRAINING data
        is_backtester = setup_backtester(None, engine_type=backtest_engine_type, config=backtest_config)
        is_backtester.data = train_data # Crucial: Set training data
        is_evaluator = MultiObjectiveEvaluator(is_backtester, objectives=objectives)
        
        # Initialize Population for this step
        population = Population(size=pop_size, dna_length=dna_length)
        population.initialize()
        # TODO: Option to seed population from previous WFO step's best?
        
        # Initialize adaptive rates/diversity/etc. for this step if needed
        current_mutation_configs_step = [m_cfg.copy() for m_cfg in mutation_configs] # Start fresh or carry over?
        current_crossover_config_step = crossover_config.copy()
        novelty_archive_step = NoveltyArchive() # Separate archive per step?
        # ... other state resets ...
        
        # --- Inner Evolutionary Loop (on Training Data) --- 
        print(f"Running evolution ({generations} generations) on training data...")
        for gen in range(generations):
            # Evaluate, Share, Select, Breed, Replace within the current population
            # This reuses the logic from basic_evolution.py but operates on `population`
            # and uses `is_evaluator` (with train_data)
            
            # Evaluate Pt
            population.evaluate_fitnesses(is_evaluator, compliance_rules)
            valid_fitness_scores_pt = {i: score for i, score in population.fitness_scores.items() if score is not None}
            if not valid_fitness_scores_pt: break # Stop if no valid parents
            
            # Adapt rates (optional, based on current pop diversity)
            if adapt_rates_enabled:
                 parent_dna = [ind.raw_dna for ind in population.individuals]
                 avg_ham = calculate_average_hamming_distance(parent_dna, 100)
                 current_mutation_configs_step, current_crossover_config_step = adapt_rates(
                     avg_ham, target_diversity, current_mutation_configs_step, 
                     current_crossover_config_step, adaptive_rate_config)
            
            # Apply Sharing (optional)
            selection_fitness = valid_fitness_scores_pt
            if sharing_enabled:
                parent_dna = [ind.raw_dna for ind in population.individuals]
                selection_fitness = apply_fitness_sharing(
                    parent_dna, valid_fitness_scores_pt, sigma_share, sharing_alpha, True)
            
            # Select Parents
            parent_indices = parent_selector_fitness(len(population), selection_fitness, pop_size)
            if not parent_indices: break

            # Create & Evaluate Offspring Qt
            offspring_chromosomes = []
            # ... (Crossover/Mutation logic using current_mutation_configs_step, current_crossover_config_step) ...
            while len(offspring_chromosomes) < pop_size: 
                # Simplified pairing for brevity
                idx1, idx2 = random.sample(parent_indices, 2)
                p1_dna, p2_dna = population[idx1].raw_dna, population[idx2].raw_dna
                o1_dna, o2_dna = p1_dna, p2_dna # Default if no crossover
                if random.random() < current_crossover_config_step.get('rate', 0.7) and len(p1_dna)==len(p2_dna):
                     o1_dna, o2_dna = crossover_op(p1_dna, p2_dna)
                mut_o1, mut_o2 = o1_dna, o2_dna
                for op, m_cfg in zip([m[0] for m in mutation_ops_config], current_mutation_configs_step):
                    mut_o1 = op(mut_o1, m_cfg.get('rate', 0.1))
                    mut_o2 = op(mut_o2, m_cfg.get('rate', 0.1))
                offspring_chromosomes.append(decode_chromosome(mut_o1))
                if len(offspring_chromosomes) < pop_size: offspring_chromosomes.append(decode_chromosome(mut_o2))

            offspring_fitness = {}
            for k, chromo in enumerate(offspring_chromosomes):
                 offspring_fitness[k] = is_evaluator.evaluate(chromo, compliance_rules)
                 
            # Combine Pt and Qt
            combined_chromos = population.individuals + offspring_chromosomes
            combined_fitness = {} 
            for k, score in valid_fitness_scores_pt.items(): combined_fitness[k] = score
            for k, score in offspring_fitness.items(): combined_fitness[len(population)+k] = score

            # Select Pt+1
            next_gen_indices = nsga2_selection(len(combined_chromos), combined_fitness, pop_size)
            if not next_gen_indices: break
            
            # Update Population
            new_individuals = [combined_chromos[idx] for idx in next_gen_indices]
            new_fitness = {ni: combined_fitness[idx] for ni, idx in enumerate(next_gen_indices)}
            population.individuals = new_individuals
            population.fitness_scores = new_fitness
            
            # Basic Gen progress print
            if (gen + 1) % 5 == 0:
                print(f"  WFO Step {i+1}, Gen {gen+1} done.")
                
        # --- End Inner Loop --- 
        
        # 4. Select Best Chromosome(s) from Final Population of this step
        print("Selecting best chromosome(s) from training step...")
        final_valid_fitness = {i: score for i, score in population.fitness_scores.items() if score is not None}
        if not final_valid_fitness:
            print("  No valid solutions found in this training step.")
            continue
            
        final_fronts = fast_non_dominated_sort(final_valid_fitness)
        if not final_fronts or not final_fronts[0]:
             print("  No non-dominated front found in this training step.")
             continue
             
        # Select the first individual from the best front (simplistic choice)
        # TODO: Implement selection based on crowding or other criteria from the front
        best_idx_this_step = final_fronts[0][0]
        best_chromosome_this_step = population.individuals[best_idx_this_step]
        all_best_chromosomes.append(best_chromosome_this_step)
        print(f"  Selected chromosome (Index {best_idx_this_step}) with fitness: {final_valid_fitness[best_idx_this_step]}")
        
        # 5. Evaluate Selected Chromosome(s) on OOS Data (test_data)
        print("Evaluating selected chromosome on OOS data...")
        oos_backtester = setup_backtester(None, engine_type=backtest_engine_type, config=backtest_config)
        oos_backtester.data = test_data # Set OOS data
        oos_evaluator = MultiObjectiveEvaluator(oos_backtester, objectives=objectives)
        oos_result = oos_evaluator.evaluate(best_chromosome_this_step, compliance_rules)
        # Wrap in BacktestingResults for consistency?
        oos_results_obj = BacktestingResults()
        oos_results_obj.stats['OOS_Fitness'] = oos_result
        # Ideally, run the *actual* backtest and store full OOS stats/trades
        # oos_results_obj = oos_backtester.run(best_chromosome_this_step, compliance_rules)
        all_oos_results.append(oos_results_obj)
        print(f"  OOS Result (Fitness Objectives): {oos_result}")
        
    # --- End WFO Loop --- 
    print("\nWalk-Forward Optimization finished.")
    
    # 6. Aggregate and Analyze OOS Results
    print("\n--- Aggregated OOS Results ---")
    if not all_oos_results:
        print("No OOS results generated.")
        return
        
    # Example: Print average OOS performance for each objective
    num_objectives = len(objectives)
    avg_oos_perf = [0.0] * num_objectives
    valid_oos_runs = 0
    for res in all_oos_results:
         oos_fitness = res.stats.get('OOS_Fitness')
         # Check if fitness is valid (not None, not worst values)
         if oos_fitness and all(f != -float('inf') and f != float('inf') for f in oos_fitness):
             valid_oos_runs += 1
             for i in range(num_objectives):
                 avg_oos_perf[i] += oos_fitness[i]
                 
    if valid_oos_runs > 0:
        avg_oos_perf = [perf / valid_oos_runs for perf in avg_oos_perf]
        print(f"Average OOS Performance across {valid_oos_runs} valid runs:")
        for i, name in enumerate(objectives):
            print(f"  {name}: {avg_oos_perf[i]:.4f}")
    else:
         print("No valid OOS runs completed successfully.")
         
    # TODO: Add more sophisticated analysis (stability, equity curve combination, etc.)
    # TODO: Save/report the collection of `all_best_chromosomes` found.

if __name__ == "__main__":
    run_walk_forward_evolution() 