"""Visualization functions for evolutionary results."""

from typing import List, Dict, Tuple, Optional
import random

# Need FitnessType definition
from nucleotide_strategy_evolution.fitness.ranking import FitnessType
# Need Chromosome definition
from nucleotide_strategy_evolution.core.structures import Chromosome

# Import plotting library
import matplotlib.pyplot as plt
# Import plotly for parallel coordinates
import plotly.graph_objects as go
import pandas as pd # Needed for parallel coordinates data structure
import numpy as np # Add numpy import
import seaborn as sns # Add seaborn for heatmaps

# Import MAP-Elites Archive for type hint
from ..population.map_elites import MapElitesArchive 

def plot_pareto_front_2d(
    population: List[Chromosome],
    fitness_scores: Dict[int, FitnessType], 
    front_indices: List[int],
    objective_names: List[str],
    title: str = "Pareto Front Visualization",
    save_path: Optional[str] = None
):
    """Creates a 2D scatter plot of the non-dominated front.

    Args:
        population: The list of Chromosome objects (needed if full pop plot desired).
        fitness_scores: Dictionary mapping individual index (relative to population list) 
                        to its fitness tuple.
        front_indices: List of indices (relative to population list) belonging to the 
                       non-dominated front (Rank 0).
        objective_names: List of names for the objectives (length must match fitness tuple).
                         The first two names are used for the X and Y axes.
        title: Title for the plot.
        save_path: If provided, saves the plot to this file path instead of showing it.
    """
    if not front_indices:
        print("Warning: Cannot plot empty front.")
        return
        
    if not fitness_scores:
        print("Warning: No fitness scores provided for plotting.")
        return

    # Get fitness values for the front individuals
    front_fitness = [fitness_scores.get(idx) for idx in front_indices]
    # Filter out None values just in case
    valid_front_fitness = [f for f in front_fitness if f is not None and len(f) >= 2]

    if not valid_front_fitness:
        print("Warning: No valid fitness scores found for front individuals.")
        return
        
    if len(objective_names) < 2:
        print("Warning: Need at least two objective names for 2D plot.")
        return
        
    # Extract X and Y values (first two objectives)
    x_values = [f[0] for f in valid_front_fitness]
    y_values = [f[1] for f in valid_front_fitness]
    
    # --- Create Plot --- 
    plt.figure(figsize=(10, 8))
    
    # Plot the non-dominated front points
    plt.scatter(x_values, y_values, c='red', marker='o', label='Pareto Front (Rank 0)', s=50, zorder=3)
    
    # Optional: Plot dominated points from the rest of the population (if desired)
    # This requires iterating through all fitness scores and plotting non-front points
    # Example:
    # all_x = []
    # all_y = []
    # for idx, fitness in fitness_scores.items():
    #     if idx not in front_indices and fitness is not None and len(fitness) >= 2:
    #         all_x.append(fitness[0])
    #         all_y.append(fitness[1])
    # if all_x:
    #     plt.scatter(all_x, all_y, c='blue', marker='.', label='Dominated Solutions', s=10, alpha=0.5, zorder=1)

    plt.xlabel(objective_names[0])
    plt.ylabel(objective_names[1])
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Pareto front plot saved to: {save_path}")
        plt.close() # Close the plot window if saving
    else:
        plt.show()

def plot_metric_history(
    history: List[float],
    metric_name: str = "Metric",
    title: Optional[str] = None,
    save_path: Optional[str] = None
):
    """Plots the history of a metric over generations.

    Args:
        history: A list of metric values, one per generation.
        metric_name: The name of the metric for the Y-axis label.
        title: The title for the plot. Defaults to "Metric History".
        save_path: If provided, saves the plot instead of showing it.
    """
    if not history:
        print("Warning: Cannot plot empty history.")
        return
        
    generations = list(range(1, len(history) + 1))
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations, history, marker='.', linestyle='-')
    plt.xlabel("Generation")
    plt.ylabel(metric_name)
    plot_title = title if title else f"{metric_name} History Over Generations"
    plt.title(plot_title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"{metric_name} history plot saved to: {save_path}")
        plt.close()
    else:
        plt.show()

def plot_parallel_coordinates(
    fitness_scores: Dict[int, FitnessType], 
    front_indices: List[int],
    objective_names: List[str],
    title: str = "Parallel Coordinates Plot (Pareto Front)",
    save_path: Optional[str] = None # Saving HTML for plotly
):
    """Creates a parallel coordinates plot for individuals on the Pareto front.

    Args:
        fitness_scores: Dictionary mapping individual index to its fitness tuple.
        front_indices: List of indices belonging to the non-dominated front (Rank 0).
        objective_names: List of names for the objectives.
        title: Title for the plot.
        save_path: If provided, saves the plot as an HTML file.
    """
    if not front_indices:
        print("Warning: Cannot plot empty front for parallel coordinates.")
        return
        
    if not fitness_scores or not objective_names:
        print("Warning: No fitness scores or objective names provided for parallel coordinates plot.")
        return
        
    # Extract fitness data for the front
    front_data = []
    for idx in front_indices:
        fitness = fitness_scores.get(idx)
        if fitness is not None and len(fitness) == len(objective_names):
            front_data.append(list(fitness))
        else:
             print(f"Warning: Skipping individual {idx} due to missing or mismatched fitness data.")

    if not front_data:
        print("Warning: No valid data points found for the front.")
        return

    # Create dimensions for the parallel coordinates plot
    dimensions = []
    # Convert data to DataFrame for easier range calculation
    df = pd.DataFrame(front_data, columns=objective_names)
    
    for i, name in enumerate(objective_names):
        # Handle potential infinite values when calculating range
        valid_col_data = df.iloc[:, i][~df.iloc[:, i].isin([float('inf'), -float('inf')])]
        data_range = [valid_col_data.min(), valid_col_data.max()] if not valid_col_data.empty else [0, 1] # Default range if all inf
        # Add slight padding to range if min == max
        if data_range[0] == data_range[1]:
             data_range[0] -= 0.5
             data_range[1] += 0.5
             
        dimensions.append(dict(
            range = data_range,
            label = name,
            values = df.iloc[:, i].tolist() # Use original data with potential inf values
        ))

    # Create the plot
    fig = go.Figure(data=
        go.Parcoords(
            line = dict(color = 'blue'), # Can add color scale based on an objective later
            dimensions = list(dimensions)
        )
    )
    
    fig.update_layout(
        title=title,
    )
    
    if save_path:
        if not save_path.lower().endswith('.html'):
             save_path += '.html'
        fig.write_html(save_path)
        print(f"Parallel coordinates plot saved to: {save_path}")
    else:
        fig.show()

def plot_map_elites_heatmap(
    map_archive: MapElitesArchive,
    behavior_dim_indices: Tuple[int, int] = (0, 1), # Which two behavior dimensions to plot
    behavior_dim_names: Optional[List[str]] = None,
    title: str = "MAP-Elites Archive Heatmap",
    save_path: Optional[str] = None
):
    """Creates a 2D heatmap visualization of the MAP-Elites archive.

    Args:
        map_archive: The populated MapElitesArchive object.
        behavior_dim_indices: Tuple specifying the two behavior dimensions to use
                              for the X and Y axes of the heatmap.
        behavior_dim_names: Optional list of names for the behavior dimensions. If provided,
                            used for axis labels. Must match the archive's dimensions.
        title: Title for the plot.
        save_path: If provided, saves the plot to this file path.
    """
    if map_archive.behavior_dims < 2:
        print("Warning: Need at least 2 behavior dimensions for heatmap plot.")
        return
        
    if not 0 <= behavior_dim_indices[0] < map_archive.behavior_dims or \
       not 0 <= behavior_dim_indices[1] < map_archive.behavior_dims or \
       behavior_dim_indices[0] == behavior_dim_indices[1]:
           raise ValueError(f"Invalid behavior_dim_indices {behavior_dim_indices} for {map_archive.behavior_dims} dimensions.")
           
    if behavior_dim_names and len(behavior_dim_names) != map_archive.behavior_dims:
         print("Warning: Length of behavior_dim_names does not match archive dimensions. Ignoring names.")
         behavior_dim_names = None

    dim_x, dim_y = behavior_dim_indices
    num_bins_x = map_archive.bins_per_dim[dim_x]
    num_bins_y = map_archive.bins_per_dim[dim_y]
    
    # Initialize grid with NaN or a suitable value for empty cells
    heatmap_data = np.full((num_bins_y, num_bins_x), np.nan)

    # Populate the heatmap grid with fitness values
    for key, (fitness, _, _) in map_archive.grid.items():
        if len(key) == map_archive.behavior_dims: # Ensure key is valid
            bin_x = key[dim_x]
            bin_y = key[dim_y]
            # Check if this key corresponds to the slice we are plotting
            # (Assumes other dimensions are implicitly fixed or we average/max over them - simple approach first)
            # For a true 2D slice, we might need to filter keys more strictly if dims > 2.
            # For now, project all filled cells onto the chosen 2D plane.
            if 0 <= bin_x < num_bins_x and 0 <= bin_y < num_bins_y:
                 current_val = heatmap_data[bin_y, bin_x]
                 # If cell already has a value (due to projection from higher dims),
                 # decide whether to overwrite (e.g., take max fitness)
                 if np.isnan(current_val) or \
                    (not map_archive.minimize and fitness > current_val) or \
                    (map_archive.minimize and fitness < current_val): 
                      heatmap_data[bin_y, bin_x] = fitness

    # --- Create Heatmap Plot --- 
    plt.figure(figsize=(10, 8))
    
    # Choose colormap (e.g., 'viridis', 'plasma', 'magma')
    cmap = 'viridis' if not map_archive.minimize else 'viridis_r'
    
    # Handle cases with all NaNs to avoid colorbar errors
    valid_data = heatmap_data[~np.isnan(heatmap_data)]
    vmin = valid_data.min() if len(valid_data) > 0 else 0
    vmax = valid_data.max() if len(valid_data) > 0 else 1
    if vmin == vmax and len(valid_data) > 0: # Adjust if all values are the same
         vmin -= 0.5
         vmax += 0.5
         
    ax = sns.heatmap(heatmap_data, cmap=cmap, annot=False, fmt=".2f", 
                     linewidths=.5, linecolor='gray', cbar=True,
                     vmin=vmin, vmax=vmax, # Set explicit color limits
                     cbar_kws={'label': 'Fitness'})
                     
    ax.invert_yaxis() # Often makes sense for grid representations

    # Set axis labels using behavior bounds and names
    x_label = f"Behavior Dim {dim_x}" 
    y_label = f"Behavior Dim {dim_y}"
    if behavior_dim_names:
        x_label = behavior_dim_names[dim_x]
        y_label = behavior_dim_names[dim_y]
        
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    # Customize ticks to show bin ranges (optional, can get crowded)
    # Use the precomputed bin edges from the archive
    x_ticks = map_archive._bin_edges[dim_x]
    y_ticks = map_archive._bin_edges[dim_y]
    plt.xticks(ticks=np.arange(num_bins_x + 1), 
               labels=[f'{t:.1f}' for t in x_ticks], 
               rotation=45, ha='right')
    plt.yticks(ticks=np.arange(num_bins_y + 1), 
               labels=[f'{t:.1f}' for t in y_ticks])

    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"MAP-Elites heatmap saved to: {save_path}")
        plt.close()
    else:
        plt.show()

# TODO PH3: Implement other visualizations (e.g., gene expression heatmaps)

# --- Example Usage ---
if __name__ == '__main__':
    # Example Data (similar to ranking example)
    mock_fitness_scores = {
        0: (10, 1), 1: (8, 3), 2: (8, 3), 3: (6, 5), 4: (4, 6),
        5: (2, 7), 6: (9, 2), 7: (7, 4), 8: (5, 5.5), 9: (3, 6.5)
    }
    # Assume a dummy population list corresponding to indices
    mock_population = [Chromosome(raw_dna=DNASequence(str(i))) for i in range(10)] 
    
    # Manually define front 0 indices from ranking example
    front0_indices = [0, 6]
    objective_names = ["Profit", "-Drawdown"]
    
    print("Plotting example Pareto Front...")
    plot_pareto_front_2d(
        population=mock_population, # Not used in basic plot, but pass for completeness
        fitness_scores=mock_fitness_scores,
        front_indices=front0_indices,
        objective_names=objective_names,
        title="Example Pareto Front (Profit vs -Drawdown)"
    )
    
    # Example saving
    # plot_pareto_front_2d(... , save_path="pareto_example.png") 
    
    print("\nPlotting example metric history...")
    mock_history = [random.random() * 10 + 50 for _ in range(20)] # Simulate 20 generations
    plot_metric_history(mock_history, metric_name="Average Fitness", title="Example Fitness History")
    
    print("\nPlotting example parallel coordinates...")
    # Use the multi-objective fitness data
    plot_parallel_coordinates(
        fitness_scores=mock_fitness_scores, 
        front_indices=front0_indices, # Plotting front 0
        objective_names=objective_names, 
        title="Example Parallel Coordinates (Front 0)"
    )
    # Example saving
    # plot_parallel_coordinates(... , save_path="parallel_coords_example.html") 

    print("\nPlotting example MAP-Elites Heatmap...")
    # Create dummy MAP-Elites archive
    dummy_archive = MapElitesArchive(
        behavior_dims=2,
        bins_per_dim=[10, 10],
        behavior_bounds=[(0, 1), (0, 100)], # e.g., WinRate vs NumTrades
        fitness_objective_index=0, minimize=False
    )
    # Populate with some random data
    for i in range(50): # Add 50 random elites
         mock_chrom = Chromosome(raw_dna=DNASequence(f"DNA_{i}"))
         mock_fitness = random.random() * 100
         mock_behavior = [
             random.random(), # Dim 0 (0 to 1)
             random.random() * 100 # Dim 1 (0 to 100)
         ]
         dummy_archive.add_to_archive(mock_chrom, mock_fitness, mock_behavior)

    plot_map_elites_heatmap(
        map_archive=dummy_archive,
        behavior_dim_names=["Win Rate", "Number of Trades"],
        title="Example MAP-Elites Grid (Win Rate vs Num Trades)"
    )
    # Example saving
    # plot_map_elites_heatmap(..., save_path="map_elites_example.png")