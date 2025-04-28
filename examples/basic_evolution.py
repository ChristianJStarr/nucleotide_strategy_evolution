"""Basic example script demonstrating a simple multi-objective evolutionary loop (NSGA-II)."""

import random
import os
import sys
import functools
from typing import List, Tuple, Dict, Optional

# Add project root to sys.path to allow imports from the package
# This assumes the script is run from the project root or the examples folder
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from nucleotide_strategy_evolution.population import Population
from nucleotide_strategy_evolution.population.island import IslandModel
from nucleotide_strategy_evolution.population.selection import get_selection_operator, nsga2_selection
from nucleotide_strategy_evolution.population.diversity import calculate_average_hamming_distance, apply_fitness_sharing, NoveltyArchive, calculate_novelty_score
from nucleotide_strategy_evolution.population.behavior import characterize_behavior, BehaviorVector
from nucleotide_strategy_evolution.operators.crossover import get_crossover_operator
from nucleotide_strategy_evolution.operators.mutation import get_mutation_operator
from nucleotide_strategy_evolution.fitness.evaluation import MultiObjectiveEvaluator
from nucleotide_strategy_evolution.fitness.ranking import fast_non_dominated_sort, calculate_crowding_distance, FitnessType
from nucleotide_strategy_evolution.backtesting.interface import setup_backtester
from nucleotide_strategy_evolution.utils.config_loader import load_config
from nucleotide_strategy_evolution.core.structures import Chromosome, DNASequence
from nucleotide_strategy_evolution.visualization.plotting import (
    plot_pareto_front_2d, 
    plot_metric_history,
    plot_parallel_coordinates
)
from nucleotide_strategy_evolution.operators.adaptive import adapt_rates

# --- Configuration ---
# Use relative paths assuming script is run from project root or examples folder
EVO_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'evolution_params.yaml')
RULES_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'compliance_rules.yaml')
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'sample_data.csv') # Placeholder

# --- Main Evolution Function ---

def run_moo_evolution():
    """Runs a multi-objective genetic algorithm loop using NSGA-II, potentially with Island Model."""
    
    # 1. Load Configuration
    try:
        evo_config = load_config(EVO_CONFIG_PATH)
        compliance_rules = load_config(RULES_CONFIG_PATH)
        print("Configuration loaded.")
    except Exception as e:
        print(f"Error loading/parsing configuration: {e}")
        return
        
    # Extract parameters 
    pop_size = evo_config.get('population', {}).get('size', 100)
    dna_length = evo_config.get('population', {}).get('dna_length', 300)
    generations = evo_config.get('evolution', {}).get('generations', 50)
    crossover_config = evo_config.get('operators', {}).get('crossover', {'type': 'single_point', 'rate': 0.7})
    mutation_configs = evo_config.get('operators', {}).get('mutation', []) 
    if not isinstance(mutation_configs, list):
        mutation_configs = [{'type': 'point_mutation', 'rate': 0.1}]
    selection_config = evo_config.get('selection', {'method': 'nsga2'}) # Use nsga2 for parent selection
    objectives = evo_config.get('fitness', {}).get('objectives', ['net_profit', '-max_drawdown'])
    # Load diversity config
    diversity_config = evo_config.get('diversity', {})
    sharing_config = diversity_config.get('fitness_sharing', {'enabled': False})
    sharing_enabled = sharing_config.get('enabled', False)
    sigma_share = sharing_config.get('sigma_share', 10.0)
    sharing_alpha = sharing_config.get('alpha', 1.0)
    # Load Novelty Search config (placeholders for now)
    novelty_config = diversity_config.get('novelty_search', {'enabled': False})
    novelty_enabled = novelty_config.get('enabled', False) # Keep disabled by default
    novelty_k = novelty_config.get('k_neighbors', 10)
    novelty_archive_capacity = novelty_config.get('archive_capacity', 100)
    novelty_add_threshold = novelty_config.get('add_threshold', 1.0) # Example threshold
    # Load Adaptive Rate config
    adaptive_rate_config = evo_config.get('adaptive_rates', {'enabled': False})
    adapt_rates_enabled = adaptive_rate_config.get('enabled', False)
    target_diversity = adaptive_rate_config.get('target_hamming_distance', 15.0)
    # Load Island Model Config
    island_config = evo_config.get('island_model', {'enabled': False})
    island_enabled = island_config.get('enabled', False)
    # Get backtest engine type
    backtest_engine_type = evo_config.get('backtesting', {}).get('engine', 'backtesting.py') # Default to real one
    backtest_config = evo_config.get('backtesting', {}).get('config', {}) # Pass config to backtester
    
    # --- Override Selection Objective if Novelty Enabled --- 
    # If novelty search is on, it typically replaces or heavily influences selection
    selection_objective = selection_config.get('selection_objective', 'fitness').lower()
    if novelty_enabled and selection_objective != 'novelty':
        print("INFO: Novelty search enabled, overriding selection objective to 'novelty'.")
        selection_objective = 'novelty'
        
    # 2. Setup Dependencies
    print(f"Setting up backtester ({backtest_engine_type}) and MOO evaluator...")
    backtester = setup_backtester(DATA_PATH, engine_type=backtest_engine_type, config=backtest_config) 
    evaluator = MultiObjectiveEvaluator(backtester, objectives=objectives)
    
    crossover_op = get_crossover_operator(crossover_config)
    mutation_ops = []
    for m_config in mutation_configs:
        try:
            op_func = get_mutation_operator(m_config)
            rate = m_config.get('rate')
            if rate is not None:
                mutation_ops.append((op_func, rate))
            else:
                print(f"Warning: Mutation config missing 'rate' for type '{m_config.get('type')}'. Skipping.")
        except ValueError as e:
            print(f"Warning: Error getting mutation operator for config {m_config}: {e}. Skipping.")
            
    # Determine selection objective
    parent_selector_fitness = get_selection_operator(selection_config) # For fitness-based
    # Need basic tournament selector for novelty
    tournament_k = selection_config.get('k', 3)
    parent_selector_novelty = functools.partial(tournament_selection, k=tournament_k)
    print(f"Using parent selection objective: {selection_objective}")
    
    # Store current operator configs which hold the rates
    current_crossover_config = crossover_config.copy()
    current_mutation_configs = [m_cfg.copy() for m_cfg in mutation_configs]
    
    # 3. Initialize Population or Island Model
    if island_enabled:
        print(f"Initializing Island Model...")
        island_model = IslandModel(config=island_config, global_config=evo_config)
        # Evaluate initial islands
        print("\nEvaluating initial islands...")
        island_model.evaluate_all(evaluator, compliance_rules)
        population_manager = island_model # Use island model to manage populations
    else:
        print(f"Initializing single population (Size: {pop_size}, DNA Length: {dna_length})...")
        population = Population(size=pop_size, dna_length=dna_length)
        population.initialize()
        # Evaluate initial population
        print("\nEvaluating initial population...")
        population.evaluate_fitnesses(evaluator, compliance_rules)
        population_manager = population # Use single population directly
    
    # --- Initialize Novelty Archive --- 
    novelty_archive = NoveltyArchive(capacity=novelty_archive_capacity)
    # Store behavior vectors per individual {index: bv}
    population_behavior_vectors: Dict[int, Optional[BehaviorVector]] = {}
    # Note: In island model, each island might need its own archive or share a global one
    # For simplicity, using one global archive here.

    # --- Initial Population Evaluation ---
    print("\nEvaluating initial population...")
    if island_enabled:
        island_model.evaluate_all(evaluator, compliance_rules)
    else:
        population.evaluate_fitnesses(evaluator, compliance_rules)
    
    # --- Evolution Loop (t = 0 to Generations-1) ---
    print(f"\nStarting MOO evolution ({generations} generations)...")
    
    # Store history for plotting
    avg_hamming_history: List[float] = []

    for gen in range(generations):
        print(f"\n--- Generation {gen + 1}/{generations} ---")
        
        # --- Evolution within each island (or single population) --- 
        num_populations = len(population_manager) if island_enabled else 1
        all_dna_sequences: List[DNASequence] = [] # Collect all DNA for global diversity
        all_behavior_vectors: List[Optional[BehaviorVector]] = [] # Collect all BVs for novelty calc
        all_fitness_scores_orig: Dict[int, FitnessType] = {} # Global index needed?
        all_individuals: List[Chromosome] = []
        global_idx_offset = 0
        
        for island_idx in range(num_populations):
            current_pop = population_manager.get_island(island_idx) if island_enabled else population_manager
            print(f"-- Processing Population/Island {island_idx} --")

            # Report Diversity
            parent_dna_sequences = [ind.raw_dna for ind in current_pop.individuals]
            all_dna_sequences.extend(parent_dna_sequences) # Collect for overall adaptation
            avg_hamming_dist = calculate_average_hamming_distance(parent_dna_sequences, sample_size=100)
            print(f"  Avg Hamming Distance: {avg_hamming_dist:.2f}")
            
            # --- Calculate Behavior Vectors & Novelty --- 
            print("  Calculating behavior vectors & novelty...")
            current_behavior_vectors: List[Optional[BehaviorVector]] = []
            current_novelty_scores: Dict[int, float] = {} # Map local island index to novelty
            
            for i, ind in enumerate(current_pop.individuals):
                 # !!! This is inefficient - requires re-running backtest or storing results !!!
                 # --- Re-run backtest temporarily to get results for behavior --- 
                 # TODO: Refactor evaluation process to store/return full results
                 temp_results = evaluator.backtester.run(ind, compliance_rules)
                 bv = characterize_behavior(temp_results)
                 # Placeholder: Simulate behavior based on fitness or random
                 current_behavior_vectors.append(bv)
                 all_behavior_vectors.append(bv)
                 all_individuals.append(ind) # Collect all individuals
                 # Store original fitness with global index
                 global_idx = global_idx_offset + i
                 all_fitness_scores_orig[global_idx] = bv
                 
            # Calculate novelty after getting all BVs for this population
            if novelty_enabled:
                 for i, bv in enumerate(current_behavior_vectors):
                     novelty = calculate_novelty_score(bv, current_behavior_vectors, novelty_archive, k=novelty_k)
                     current_novelty_scores[i] = novelty
                     # Maybe add to archive based on threshold?
                     if novelty > novelty_add_threshold:
                          novelty_archive.add_to_archive(bv)
                 avg_novelty = sum(current_novelty_scores.values()) / len(current_novelty_scores) if current_novelty_scores else 0
                 print(f"  Avg Novelty Score: {avg_novelty:.4f} (Archive size: {len(novelty_archive)})")

            global_idx_offset += len(current_pop) # Update offset for next island

            # Get original fitness scores for selection
            current_fitness_scores = {i: score for i, score in current_pop.fitness_scores.items() if score is not None}
            if not current_fitness_scores:
                print(f"  Warning: No valid fitness scores found. Skipping evolution steps for this population.")
                continue

            # Apply Fitness Sharing (if enabled)
            selection_fitness_scores = current_fitness_scores # Use original if sharing disabled
            if sharing_enabled:
                try:
                    selection_fitness_scores = apply_fitness_sharing(
                        dna_sequences=parent_dna_sequences,
                        original_fitness=current_fitness_scores,
                        sigma_share=sigma_share,
                        alpha=sharing_alpha,
                        is_multi_objective=True 
                    )
                except Exception as share_err:
                     print(f"  Warning: Error during fitness sharing: {share_err}. Using original fitness.")
                     selection_fitness_scores = current_fitness_scores

            # 1. Select Parents 
            print(f"Selecting parents based on {selection_objective}...")
            if selection_objective == 'novelty': # Assumes novelty_enabled check already done
                if not current_novelty_scores: # Check if novelty was calculated
                    print("  Warning: Novelty scores not available, falling back to fitness selection.")
                    # Fallback to fitness-based selection
                    parent_indices = parent_selector_fitness(
                        population_size=len(current_pop.individuals),
                        fitness_scores=selection_fitness_scores, 
                        num_selections=current_pop.target_size
                    )
                else:
                    parent_indices = parent_selector_novelty(
                        population_size=len(current_pop.individuals),
                        fitness_scores=current_novelty_scores, # Pass novelty scores here
                        num_selections=current_pop.target_size
                    )
            else: # Default to fitness-based selection
                if selection_objective == 'novelty' and not novelty_enabled:
                    print("  Warning: Novelty selection requested but novelty search is disabled. Using fitness.")
                parent_indices = parent_selector_fitness(
                    population_size=len(current_pop.individuals),
                    fitness_scores=selection_fitness_scores, # Use potentially shared fitness scores
                    num_selections=current_pop.target_size
                )
                
            if not parent_indices:
                print("  Warning: Selection returned no parents. Skipping offspring generation.")
                continue

            # 2. Create Offspring Population Qt
            print("Creating offspring...")
            offspring_chromosomes: List[Chromosome] = []
            
            # Get the operator *functions* once
            crossover_op_func = get_crossover_operator(current_crossover_config)
            mutation_op_funcs = []
            for m_cfg in current_mutation_configs:
                try:
                    mutation_op_funcs.append(get_mutation_operator(m_cfg))
                except ValueError as e:
                    print(f" Warning: Skipping mutation type {m_cfg.get('type')}: {e}")

            # Get parent indices for pairing
            num_parents_for_pairing = len(parent_indices)
            parent_pool_indices = parent_indices[:] # Copy indices
            if num_parents_for_pairing < 2:
                 print("  Warning: Not enough parents in pool for crossover.")
                 parent_pool_indices = parent_indices * 2 # Repeat if needed

            current_offspring_count = 0
            while current_offspring_count < current_pop.target_size:
                # Select 2 parents, allow replacement if necessary by refilling pool
                if len(parent_pool_indices) < 2:
                    parent_pool_indices = parent_indices[:] # Refill pool
                    if len(parent_pool_indices) < 2: break # Exit if still not enough
                    
                p1_idx = parent_pool_indices.pop(random.randrange(len(parent_pool_indices)))
                # Ensure p2 is different if possible from remaining pool
                if not parent_pool_indices:
                     pool_for_p2 = parent_indices[:] # Use full list if pool empty
                     if p1_idx in pool_for_p2: pool_for_p2.remove(p1_idx)
                else:
                     pool_for_p2 = parent_pool_indices
                if not pool_for_p2: # Only one parent selected/available
                     p2_idx = p1_idx # Use same parent if absolutely necessary
                else:
                     p2_idx = pool_for_p2.pop(random.randrange(len(pool_for_p2))) 
                     # Put p2 back if it wasn't from the main pool originally (it was just borrowed)
                     if parent_pool_indices is not pool_for_p2: parent_pool_indices.append(p2_idx)

                parent1_dna = current_pop[p1_idx].raw_dna
                parent2_dna = current_pop[p2_idx].raw_dna
                
                # Crossover using current rate from config
                current_crossover_rate = current_crossover_config.get('rate', 0.7)
                if random.random() < current_crossover_rate:
                    if len(parent1_dna) == len(parent2_dna):
                        # Pass swap_prob if uniform crossover
                        if current_crossover_config.get('type') == 'uniform':
                            o1_dna, o2_dna = crossover_op_func(parent1_dna, parent2_dna, swap_prob=current_crossover_config.get('swap_prob', 0.5))
                        else:
                            o1_dna, o2_dna = crossover_op_func(parent1_dna, parent2_dna)
                    else:
                        o1_dna, o2_dna = parent1_dna, parent2_dna # Skip if lengths differ
                else:
                    o1_dna, o2_dna = parent1_dna, parent2_dna 
                
                # Mutation - Apply using current rates from configs
                mutated_o1 = o1_dna
                mutated_o2 = o2_dna
                for op_func, m_cfg in zip(mutation_op_funcs, current_mutation_configs):
                    rate = m_cfg.get('rate', 0.1)
                    mutated_o1 = op_func(mutated_o1, rate)
                    mutated_o2 = op_func(mutated_o2, rate)
                    
                offspring_chromosomes.append(decode_chromosome(mutated_o1))
                current_offspring_count += 1
                if current_offspring_count < current_pop.target_size:
                    offspring_chromosomes.append(decode_chromosome(mutated_o2))
                    current_offspring_count += 1
                    
            if len(offspring_chromosomes) != current_pop.target_size:
                 print(f"  Warning: Generated {len(offspring_chromosomes)} offspring, expected {current_pop.target_size}.")

            # 3. Evaluate Fitness of Offspring Qt
            offspring_fitness_scores: Dict[int, FitnessType] = {}
            for i, chromo in enumerate(offspring_chromosomes):
                try:
                    fitness_value = evaluator.evaluate(chromo, compliance_rules)
                    offspring_fitness_scores[i] = fitness_value
                except Exception as e:
                     print(f"  Error evaluating offspring {i}: {e}")
                     worst_fitness = tuple(float('inf') if min_ else -float('inf') for _, min_ in evaluator.parsed_objectives)
                     offspring_fitness_scores[i] = worst_fitness
                 
            # 4. Combine Parent (Pt) and Offspring (Qt) -> Rt
            combined_chromosomes = current_pop.individuals + offspring_chromosomes
            combined_fitness_scores: Dict[int, FitnessType] = {}
            parent_offset = len(current_pop.individuals)
            for i, score in current_fitness_scores.items(): 
                combined_fitness_scores[i] = score
            for i, score in offspring_fitness_scores.items():
                 combined_fitness_scores[parent_offset + i] = score
                 
            # 5. Select Next Generation Pt+1 from Rt (Size N)
            combined_fronts = fast_non_dominated_sort(combined_fitness_scores)
            next_gen_indices: List[int] = []
            front_num = 0
            while len(next_gen_indices) < current_pop.target_size and front_num < len(combined_fronts):
                current_front_combined_indices = combined_fronts[front_num]
                if not current_front_combined_indices: 
                    front_num += 1
                    continue
                remaining_slots = current_pop.target_size - len(next_gen_indices)
                if len(current_front_combined_indices) <= remaining_slots:
                    next_gen_indices.extend(current_front_combined_indices)
                else:
                    distances = calculate_crowding_distance(combined_fitness_scores, current_front_combined_indices)
                    sorted_front = sorted(current_front_combined_indices, key=lambda idx: distances.get(idx, 0.0), reverse=True)
                    next_gen_indices.extend(sorted_front[:remaining_slots])
                front_num += 1
                
            # 6. Update Population object for the next iteration
            if len(next_gen_indices) != current_pop.target_size:
                 print(f"  Warning: Selected {len(next_gen_indices)} for next generation, expected {current_pop.target_size}.")
            next_gen_individuals: List[Chromosome] = []
            next_gen_fitness_scores: Dict[int, FitnessType] = {}
            for i, combined_idx in enumerate(next_gen_indices):
                next_gen_individuals.append(combined_chromosomes[combined_idx])
                next_gen_fitness_scores[i] = combined_fitness_scores[combined_idx] 
            current_pop.individuals = next_gen_individuals
            current_pop.fitness_scores = next_gen_fitness_scores

        # --- Adapt Rates (After processing all islands for the generation) --- 
        if adapt_rates_enabled:
            overall_avg_hamming = calculate_average_hamming_distance(all_dna_sequences, sample_size=200)
            print(f"\nOverall Avg Hamming: {overall_avg_hamming:.2f} (Target: {target_diversity:.2f})")
            avg_hamming_history.append(overall_avg_hamming) 
            
            # Adapt rates *in-place* by passing the current config dicts/lists
            # The function returns potentially new dicts/lists (though modify in place might be ok too)
            current_mutation_configs, current_crossover_config = adapt_rates(
                current_diversity=overall_avg_hamming,
                target_diversity=target_diversity,
                current_mutation_configs=current_mutation_configs, # Pass list of dicts
                current_crossover_config=current_crossover_config, # Pass dict
                adapt_config=adaptive_rate_config
            )
            print(f"  Adapted Crossover Config: {current_crossover_config}")
            print(f"  Adapted Mutation Configs: {current_mutation_configs}")
            
        else:
            # Still record average hamming distance if adaptation disabled
            overall_avg_hamming = calculate_average_hamming_distance(all_dna_sequences, sample_size=200)
            avg_hamming_history.append(overall_avg_hamming)
            
        # --- Migration Step (if Island Model enabled) --- 
        if island_enabled and (gen + 1) % island_model.migration_frequency == 0:
            island_model.migrate()
            # Note: Migrated individuals have fitness set to None, 
            # they will be evaluated in the next generation's evaluate_all call.
            
    # --- End of Evolution --- 
    print("\nMOO Evolution finished.")
    
    # Report the final results (e.g., best front from a specific island or combined)
    # For now, just report front 0 of island 0 if using islands
    final_population_to_report = island_model.get_island(0) if island_enabled else population_manager
    final_fitness_scores = {i: score for i, score in final_population_to_report.fitness_scores.items() if score is not None}
    if final_fitness_scores:
        final_fronts = fast_non_dominated_sort(final_fitness_scores)
        if final_fronts:
            final_best_front_indices = final_fronts[0]
            print(f"Final Pareto Front (Rank 0, Size: {len(final_best_front_indices)}):")
            # Store fitness tuples for the front for plotting
            final_front_fitness_map = {idx: final_fitness_scores[idx] for idx in final_best_front_indices}
            
            for i, idx in enumerate(final_best_front_indices):
                 chromo = final_population_to_report.individuals[idx]
                 fitness_str = str(final_population_to_report.get_fitness(idx))
                 print(f"  - Solution {i} (Index {idx}): Fitness={fitness_str}, DNA='{str(chromo.raw_dna)[:30]}...'")
                 
            # --- Visualize Final Results --- 
            # Visualize the final front (2D)
            if len(objectives) >= 2:
                try:
                    plot_pareto_front_2d(
                        population=final_population_to_report.individuals,
                        fitness_scores=final_fitness_scores, # Use all scores for potential dominated plotting
                        front_indices=final_best_front_indices,
                        objective_names=objectives, 
                        title=f"Final Pareto Front (Generation {generations})"
                        # save_path=f"pareto_gen_{generations}.png" # Example save
                    )
                except Exception as plot_err:
                     print(f"Error generating Pareto plot: {plot_err}")
            
            # Visualize the final front (Parallel Coordinates)
            if len(objectives) > 1:
                 try:
                     plot_parallel_coordinates(
                         fitness_scores=final_front_fitness_map, # Only plot front members
                         front_indices=list(final_front_fitness_map.keys()), # Pass indices relative to the map
                         objective_names=objectives,
                         title=f"Final Pareto Front Parallel Coordinates (Generation {generations})"
                         # save_path=f"parcoords_gen_{generations}.html" # Example save
                     )
                 except Exception as plot_err:
                     print(f"Error generating Parallel Coordinates plot: {plot_err}")
                 
        else:
             print("No non-dominated solutions found in the final population.")
    else:
        print("No valid solutions found in the final population.")

    # --- Plotting History ---
    if avg_hamming_history:
        try:
            plot_metric_history(
                history=avg_hamming_history,
                metric_name="Average Hamming Distance",
                title="Genotypic Diversity Over Generations"
                # save_path="hamming_history.png" # Example save
            )
        except Exception as plot_err:
            print(f"Error plotting Hamming distance history: {plot_err}")

if __name__ == "__main__":
    run_moo_evolution() 