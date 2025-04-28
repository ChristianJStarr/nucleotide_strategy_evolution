"""Functions for analyzing gene importance and contribution."""

import pandas as pd
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional
import numpy as np # For correlation calculation

from ..core.structures import Chromosome, Gene
from ..fitness.ranking import FitnessType # For fitness type hint
from ..encoding import decode_chromosome # May need decoder if input is DNA

def analyze_gene_frequency(
    population: List[Chromosome], 
    count_specific_params: bool = False
) -> Dict[str, int]:
    """Calculates the frequency of different gene types or specific gene configurations.

    Args:
        population: List of Chromosome objects (e.g., final elites).
        count_specific_params: If True, tries to create unique keys based on 
                               gene type and key parameters (can lead to many unique keys).
                               If False, only counts based on gene_type.

    Returns:
        Dictionary mapping gene identifier (str) to its frequency count.
    """
    gene_counts: Counter = Counter()
    
    for chromosome in population:
        for gene in chromosome.genes:
            if count_specific_params:
                # Create a more specific key, e.g., "entry_rule:indicator=SMA:period=14"
                # This requires a consistent way to represent key parameters.
                # Simple example: join key items
                param_items = sorted(gene.parameters.items()) # Sort for consistency
                # Filter for potentially identifying params (heuristic)
                key_params = [f"{k}={v}" for k, v in param_items 
                              if isinstance(v, (str, int, float, bool)) and k != 'conditions'] # Exclude complex params like conditions
                identifier = f"{gene.gene_type}:{':'.join(key_params)}"
            else:
                # Simple count by gene type
                identifier = gene.gene_type
            
            gene_counts[identifier] += 1
            
    return dict(gene_counts)

def correlate_genes_with_performance(
    population: List[Chromosome],
    fitness_scores: Dict[int, FitnessType], # Map chromosome index to fitness tuple
    objective_index: int = 0,
    min_frequency: int = 2 # Minimum times a gene/param must appear to be included
) -> Optional[pd.DataFrame]:
    """Calculates correlation between gene presence/parameters and a fitness objective.

    Args:
        population: List of Chromosome objects.
        fitness_scores: Dictionary mapping the original index of the chromosome in the 
                        population list to its FitnessType tuple.
        objective_index: Which fitness objective (by index in the tuple) to correlate with.
        min_frequency: Minimum number of times a gene feature must be present/
                       have a numerical value across the population to be included.

    Returns:
        Pandas DataFrame containing correlation coefficients, or None if analysis fails.
    """
    if not population:
        print("Warning: Cannot correlate genes for an empty population.")
        return None
        
    if len(population) != len(fitness_scores):
         print("Warning: Population size and fitness_scores length mismatch. Cannot perform correlation.")
         # Match based on indices present in fitness_scores
         # indices_to_use = list(fitness_scores.keys())
         # Need a way to reliably map fitness index back to population index if they differ
         return None # Simplest for now
         
    data_for_corr: Dict[str, list] = defaultdict(list)
    target_fitness_list = []
    valid_indices = []

    # --- Extract Features and Target Fitness --- 
    gene_feature_set = set() # Track all unique gene features found

    for i, chromosome in enumerate(population):
        fitness = fitness_scores.get(i)
        
        # Check if fitness is valid and has the target objective
        if fitness is None or not isinstance(fitness, tuple) or len(fitness) <= objective_index:
            continue # Skip individuals with missing/invalid fitness
            
        target_fitness = fitness[objective_index]
        if target_fitness is None or not np.isfinite(target_fitness):
             continue # Skip non-finite fitness values
             
        valid_indices.append(i)
        target_fitness_list.append(target_fitness)
        
        # Extract features for this chromosome
        chromosome_features: Dict[str, Any] = {}
        for gene in chromosome.genes:
            # Feature: Presence of gene type
            gene_type_key = f"gene_{gene.gene_type}_present"
            chromosome_features[gene_type_key] = 1
            gene_feature_set.add(gene_type_key)
            
            # Feature: Numerical parameter values
            for param_name, param_value in gene.parameters.items():
                if isinstance(param_value, (int, float)) and np.isfinite(param_value):
                    # Create a unique key for the parameter within its gene type
                    param_key = f"gene_{gene.gene_type}_{param_name}"
                    # How to handle multiple genes of same type with same param?
                    # Option 1: Average -> Requires tracking counts
                    # Option 2: Take first -> Simple but loses info
                    # Option 3: Sum -> Might not make sense
                    # Taking first for simplicity now:
                    if param_key not in chromosome_features:
                         chromosome_features[param_key] = param_value
                    gene_feature_set.add(param_key)
                    
        # Store features for this valid individual
        for feature in gene_feature_set:
            data_for_corr[feature].append(chromosome_features.get(feature, 0)) # Use 0 if feature absent
            
    # Ensure all feature lists have the same length (number of valid individuals)
    num_valid_individuals = len(target_fitness_list)
    if num_valid_individuals < 2:
         print("Warning: Need at least 2 valid individuals for correlation analysis.")
         return None
         
    for feature in gene_feature_set:
         if len(data_for_corr[feature]) < num_valid_individuals:
              # Pad missing feature occurrences for individuals processed earlier
              padding = [0] * (num_valid_individuals - len(data_for_corr[feature]))
              data_for_corr[feature].extend(padding)
              
    # Add target fitness to the dictionary
    data_for_corr[f'fitness_obj_{objective_index}'] = target_fitness_list
    
    # --- Create DataFrame and Filter Features --- 
    try:
        df = pd.DataFrame(data_for_corr, index=valid_indices)
    except ValueError as e:
         print(f"Error creating DataFrame for correlation: {e}")
         # This might happen if lists have different lengths despite padding logic
         # Print lengths for debugging:
         for k, v in data_for_corr.items(): print(f"  {k}: {len(v)}")
         return None

    # Filter columns (features) based on minimum frequency/variance
    cols_to_keep = []
    target_col = f'fitness_obj_{objective_index}'
    for col in df.columns:
        if col == target_col: continue
        # Check frequency (for binary present/absent features)
        if col.endswith('_present'):
             frequency = df[col].sum()
             if frequency >= min_frequency and frequency < len(df): # Exclude if always present/absent
                 cols_to_keep.append(col)
        # Check variance (for numerical features)
        elif pd.api.types.is_numeric_dtype(df[col]):
             if df[col].nunique() > 1 and df[col].count() >= min_frequency: # Check non-zero std dev implicitly
                 cols_to_keep.append(col)
                 
    if not cols_to_keep:
        print("Warning: No features met minimum frequency/variance criteria for correlation.")
        return None
        
    df_filtered = df[cols_to_keep + [target_col]]

    # --- Calculate Correlation --- 
    try:
        correlations = df_filtered.corr(method='pearson')[target_col].drop(target_col)
        # Could use spearman for non-linear relationships, or point-biserial for binary features
        
        # Convert to DataFrame for better output
        corr_df = correlations.reset_index()
        corr_df.columns = ['Feature', 'Correlation']
        corr_df = corr_df.sort_values(by='Correlation', key=abs, ascending=False)
        
        print(f"Correlation analysis complete for objective {objective_index}.")
        return corr_df
        
    except Exception as e:
        print(f"Error calculating correlations: {e}")
        return None 