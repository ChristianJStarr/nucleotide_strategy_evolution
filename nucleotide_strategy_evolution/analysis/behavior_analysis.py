"""Functions for analyzing behavioral diversity using dimensionality reduction."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Sequence

# Need behavior vectors, potentially associated fitness/chromosome info
from ..population.behavior import BehaviorVector
from ..core.structures import Chromosome
from ..fitness.ranking import FitnessType

# Import dimensionality reduction techniques
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Import plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_behavior_pca(
    behavior_vectors: List[Optional[BehaviorVector]],
    n_components: int = 2,
    fitness_values: Optional[List[Optional[FitnessType]]] = None, # Optional fitness for coloring
    fitness_index: int = 0 # Which fitness objective to use for color
) -> Optional[pd.DataFrame]:
    """Performs PCA on behavior vectors and returns the projected components.

    Args:
        behavior_vectors: List of behavior vectors (can contain Nones).
        n_components: Number of principal components to compute.
        fitness_values: Optional list of fitness values corresponding to behavior vectors.
                        Assumes the list index matches the index in behavior_vectors.
        fitness_index: Index of the fitness objective to potentially include in the output.

    Returns:
        Pandas DataFrame with PCA components and optionally fitness, or None if fails.
    """
    # Filter out None vectors and create mapping from original index to filtered index
    valid_bvs_with_orig_idx: List[Tuple[int, BehaviorVector]] = []
    for i, bv in enumerate(behavior_vectors):
        if bv is not None:
             valid_bvs_with_orig_idx.append((i, bv))

    if not valid_bvs_with_orig_idx:
        print("Warning: No valid behavior vectors provided for PCA.")
        return None
        
    original_indices = [item[0] for item in valid_bvs_with_orig_idx]
    valid_bvs = [item[1] for item in valid_bvs_with_orig_idx]
    
    # Filter corresponding fitness values
    valid_fitness = None
    fitness_column_data = []
    if fitness_values:
         if len(fitness_values) != len(behavior_vectors):
             print("Warning: Length of fitness_values does not match behavior_vectors. Ignoring fitness.")
         else:
             valid_fitness_raw = [fitness_values[i] for i in original_indices]
             # Extract the specific objective
             for fit in valid_fitness_raw:
                 if fit is not None and isinstance(fit, Sequence) and len(fit) > fitness_index:
                     fitness_column_data.append(fit[fitness_index])
                 elif fit is not None and isinstance(fit, (int, float)) and fitness_index == 0:
                      fitness_column_data.append(float(fit)) # Handle single objective case
                 else:
                      fitness_column_data.append(np.nan) # Use NaN for missing/invalid fitness
             if len(fitness_column_data) != len(valid_bvs):
                  print("Warning: Internal error matching fitness values to valid behaviors. Ignoring fitness.")
                  fitness_column_data = []
                  
    # Prepare data for PCA
    X = np.array(valid_bvs)
    if X.shape[0] < n_components:
         print(f"Warning: Number of valid samples ({X.shape[0]}) is less than n_components ({n_components}). Returning None.")
         return None
         
    if X.shape[1] < n_components:
        print(f"Warning: Number of features ({X.shape[1]}) is less than n_components ({n_components}). Adjusting n_components.")
        n_components = X.shape[1]
        
    if n_components <= 0:
         print("Warning: No features available for PCA after filtering.")
         return None

    # Scale data before PCA
    scaler = StandardScaler()
    try:
         X_scaled = scaler.fit_transform(X)
    except ValueError as e:
         print(f"Error scaling data for PCA (perhaps constant features?): {e}")
         return None 

    pca = PCA(n_components=n_components)
    try:
        principal_components = pca.fit_transform(X_scaled)
    except Exception as e:
        print(f"Error during PCA fitting: {e}")
        return None

    # Create DataFrame
    pca_df = pd.DataFrame(data=principal_components,
                          columns=[f'PC{i+1}' for i in range(n_components)],
                          index=original_indices) # Use original index
                          
    explained_variance_ratio = pca.explained_variance_ratio_
    print(f"PCA Explained Variance Ratio ({n_components} components): {explained_variance_ratio}")
    print(f"Total Explained Variance: {explained_variance_ratio.sum():.4f}")

    # Add fitness if successfully extracted
    if fitness_column_data:
         fitness_col_name = f'FitnessObj_{fitness_index}'
         pca_df[fitness_col_name] = fitness_column_data

    return pca_df

def plot_behavior_projection(
    projection_df: pd.DataFrame,
    title: str = "Behavior Space Projection",
    color_by_column: Optional[str] = None, # e.g., 'FitnessObj_0' or 'TSNE-1' if using tSNE df
    x_axis_col: str = 'PC1', # Default to PCA, allow override for t-SNE
    y_axis_col: str = 'PC2', 
    save_path: Optional[str] = None
):
    """Plots the 2D projection (e.g., PCA or t-SNE results)."""
    if projection_df is None or projection_df.empty:
        print("Warning: Cannot plot empty projection DataFrame.")
        return
        
    if x_axis_col not in projection_df.columns or y_axis_col not in projection_df.columns:
         print(f"Warning: DataFrame missing required columns '{x_axis_col}' or '{y_axis_col}' for plot.")
         return

    plt.figure(figsize=(10, 8))
    
    hue_data = None
    cmap = 'viridis'
    if color_by_column and color_by_column in projection_df.columns:
         hue_data = projection_df[color_by_column]
         # Filter out NaNs for coloring if they exist (e.g., from fitness)
         valid_indices = ~hue_data.isna()
         if not valid_indices.all():
              print(f"Warning: NaN values found in color_by column '{color_by_column}'. Plotting only valid points.")
              projection_df = projection_df[valid_indices]
              hue_data = hue_data[valid_indices]
         print(f"Coloring plot by: {color_by_column}")
    else:
         print("No valid color_by_column specified, using default color.")
         # Need to handle case where projection_df might be empty after filtering NaNs
         if projection_df.empty:
             print("Warning: No valid data points left after filtering NaNs in color column.")
             return 

    scatter = sns.scatterplot(
        x=x_axis_col,
        y=y_axis_col,
        hue=hue_data,
        palette=cmap,
        data=projection_df,
        legend='auto' if hue_data is not None else False,
        s=50, # marker size
        alpha=0.7
    )

    plt.xlabel(x_axis_col)
    plt.ylabel(y_axis_col)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add color bar if coloring by continuous variable
    if hue_data is not None and pd.api.types.is_numeric_dtype(hue_data):
         # Check if hue_data is empty after potential filtering
         if not hue_data.empty:
             norm = plt.Normalize(hue_data.min(), hue_data.max())
             sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
             sm.set_array([])
             # Remove the legend created by scatterplot if using color bar
             if hasattr(scatter, 'legend_') and scatter.legend_ is not None: # Check legend exists
                 scatter.legend_.remove()
             try:
                 plt.colorbar(sm, label=color_by_column)
             except Exception as e:
                 print(f"Warning: Could not create colorbar: {e}")
         else:
              print("Warning: No valid numeric data for colorbar.")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Behavior projection plot saved to: {save_path}")
        plt.close()
    else:
        plt.show()

# TODO: Implement t-SNE analysis function (similar structure to PCA)
# def analyze_behavior_tsne(...) 