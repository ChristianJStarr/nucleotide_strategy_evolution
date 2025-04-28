"""
Generate example visualizations for the README.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from typing import List, Dict, Tuple

# Create path for output images
os.makedirs('images', exist_ok=True)

def generate_pareto_front():
    """Generate a sample Pareto front visualization."""
    # Create sample data points
    np.random.seed(42)
    
    # Generate Pareto front points
    n_points = 15
    x_pareto = np.linspace(0.1, 1.0, n_points)
    y_pareto = 1.0 / x_pareto  # Hyperbolic Pareto front
    
    # Add some noise to make it look more realistic
    x_pareto += np.random.normal(0, 0.01, n_points)
    y_pareto += np.random.normal(0, 0.1, n_points)
    
    # Generate dominated points
    n_dominated = 50
    x_dominated = np.random.uniform(0.1, 1.5, n_dominated)
    y_dominated = np.random.uniform(1.0, 5.0, n_dominated)
    
    # Filter dominated points to ensure they're actually dominated
    dominated_mask = (y_dominated > 1.0 / x_dominated)
    x_dominated = x_dominated[dominated_mask]
    y_dominated = y_dominated[dominated_mask]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_dominated, y_dominated, alpha=0.5, c='gray', label='Dominated Solutions')
    plt.scatter(x_pareto, y_pareto, c='red', marker='o', s=50, label='Pareto Front')
    
    plt.xlabel('Return (Profit)')
    plt.ylabel('Risk (Max Drawdown)')
    plt.title('Pareto Front: Return vs. Risk Trade-off')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Add annotation
    plt.annotate('Better Solutions', xy=(0.5, 1.5), xytext=(0.7, 2.5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
    
    plt.tight_layout()
    plt.savefig('images/pareto_front.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated Pareto front visualization")

def generate_metric_history():
    """Generate a sample metric history visualization."""
    np.random.seed(42)
    
    # Generate sample metrics over generations
    generations = 50
    
    # Average fitness - increasing trend with noise
    avg_fitness = np.linspace(50, 95, generations) + np.random.normal(0, 5, generations)
    
    # Best fitness - increasing trend with diminishing returns
    best_fitness = 100 - 50 * np.exp(-np.linspace(0, 3, generations)) + np.random.normal(0, 2, generations)
    
    # Diversity - decreasing trend
    diversity = 80 * np.exp(-np.linspace(0, 2, generations)) + np.random.normal(0, 3, generations)
    diversity = np.clip(diversity, 0, 100)
    
    # Create plot
    plt.figure(figsize=(12, 7))
    
    plt.plot(range(1, generations+1), avg_fitness, 'b-', label='Average Fitness')
    plt.plot(range(1, generations+1), best_fitness, 'r-', label='Best Fitness')
    plt.plot(range(1, generations+1), diversity, 'g-', label='Population Diversity')
    
    plt.xlabel('Generation')
    plt.ylabel('Metric Value')
    plt.title('Evolution Metrics Over Generations')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Add annotations
    plt.annotate('Initial Population', xy=(3, 60), xytext=(5, 40),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1))
    plt.annotate('Convergence', xy=(45, 95), xytext=(30, 85),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1))
    
    plt.tight_layout()
    plt.savefig('images/metrics_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated metric history visualization")

def generate_map_elites_grid():
    """Generate a sample MAP-Elites grid visualization."""
    np.random.seed(42)
    
    # Define grid dimensions
    grid_size_x, grid_size_y = 10, 10
    
    # Create a sample fitness grid with some empty cells
    # NaN represents empty cells
    fitness_grid = np.full((grid_size_x, grid_size_y), np.nan)
    
    # Fill about 60% of the grid with fitness values
    filled_cells = int(0.6 * grid_size_x * grid_size_y)
    for _ in range(filled_cells):
        x, y = random.randint(0, grid_size_x-1), random.randint(0, grid_size_y-1)
        # Higher fitness in the middle, lower at edges
        distance_from_center = np.sqrt(((x - grid_size_x/2) / grid_size_x*2)**2 + 
                                       ((y - grid_size_y/2) / grid_size_y*2)**2)
        fitness_grid[x, y] = np.random.normal(1.0 - distance_from_center, 0.1)
    
    # Create a more aesthetically pleasing colormap
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Create heat map
    ax = sns.heatmap(fitness_grid, cmap=cmap, linewidths=0.5, 
                    cbar_kws={'label': 'Fitness Score'},
                    mask=np.isnan(fitness_grid))
    
    # Customize the plot
    plt.title('MAP-Elites Grid: Performance Across Behavioral Dimensions')
    plt.xlabel('Trading Frequency (Behavior Dimension 1)')
    plt.ylabel('Risk Exposure (Behavior Dimension 2)')
    
    # Add annotations
    plt.text(2, 3, 'High-Performing\nNiche', color='white', 
             ha='center', va='center', fontweight='bold')
    plt.text(8, 8, 'Unexplored\nRegion', color='black', 
             ha='center', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('images/map_elites_grid.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated MAP-Elites grid visualization")

if __name__ == "__main__":
    print("Generating visualizations for README...")
    generate_pareto_front()
    generate_metric_history()
    generate_map_elites_grid()
    print("All visualizations generated successfully in the 'images' directory") 