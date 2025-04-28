#!/usr/bin/env python
"""
Script to automatically generate API documentation for Nucleotide Strategy Evolution.
"""

import os
import sys
import importlib
import inspect

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('..'))

# Module structure to document
MODULES = {
    'core': [
        'structures',
    ],
    'encoding': [],  # Document the module itself
    'genes': [],
    'operators': [
        'mutation',
        'crossover',
        'adaptive',
    ],
    'population': [
        'population',
        'diversity',
        'selection',
        'island',
        'map_elites',
    ],
    'fitness': [
        'evaluation',
        'ranking',
    ],
    'backtesting': [],
    'analysis': [
        'robustness',
        'gene_analysis',
        'behavior_analysis',
    ],
    'visualization': [
        'plotting',
    ],
    'utils': [],
}

RST_TEMPLATE = """
{heading}
{heading_underline}

.. automodule:: {module_path}
   :members:
   :undoc-members:
   :show-inheritance:

"""

INDEX_TEMPLATE = """
{heading}
{heading_underline}

.. toctree::
   :maxdepth: 2

{submodules}
"""

def generate_module_docs(module_name, submodules=None):
    """Generate RST files for a module and its submodules."""
    
    if not submodules:
        # Document the module itself
        module_path = f"nucleotide_strategy_evolution.{module_name}"
        output_path = f"source/api/{module_name}.rst"
        heading = f"{module_name.title()} Module"
        
        with open(output_path, 'w') as f:
            f.write(RST_TEMPLATE.format(
                heading=heading,
                heading_underline='=' * len(heading),
                module_path=module_path
            ))
        return
    
    # Create directory for submodules if needed
    os.makedirs(f"source/api/{module_name}", exist_ok=True)
    
    # Generate an index file for the module
    with open(f"source/api/{module_name}.rst", 'w') as f:
        heading = f"{module_name.title()} API"
        submodule_list = '\n'.join(f'   {module_name}/{submodule}' for submodule in submodules)
        
        f.write(INDEX_TEMPLATE.format(
            heading=heading,
            heading_underline='=' * len(heading),
            submodules=submodule_list
        ))
    
    # Generate a RST file for each submodule
    for submodule in submodules:
        module_path = f"nucleotide_strategy_evolution.{module_name}.{submodule}"
        output_path = f"source/api/{module_name}/{submodule}.rst"
        
        heading = f"{submodule.replace('_', ' ').title()}"
        
        with open(output_path, 'w') as f:
            f.write(RST_TEMPLATE.format(
                heading=heading,
                heading_underline='=' * len(heading),
                module_path=module_path
            ))

def main():
    """Main function to generate API documentation."""
    os.makedirs("source/api", exist_ok=True)
    
    for module_name, submodules in MODULES.items():
        generate_module_docs(module_name, submodules)
    
    print("API documentation generation complete.")

if __name__ == "__main__":
    main() 