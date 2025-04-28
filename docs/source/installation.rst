============
Installation
============

This page describes how to install the Nucleotide Strategy Evolution package.

Requirements
-----------

* Python 3.11 or higher
* pip (Python package installer)

Basic Installation
----------------

You can install the package directly from GitHub:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/ChristianJStarr/nucleotide_strategy_evolution.git
    cd nucleotide_strategy_evolution

    # Create and activate a virtual environment (recommended)
    python -m venv .venv
    
    # On Windows:
    .venv\Scripts\activate
    
    # On macOS/Linux: 
    # source .venv/bin/activate

    # Install the package and its core dependencies
    pip install -e .

Install Development Dependencies
------------------------------

If you plan to contribute to the project or run tests:

.. code-block:: bash

    # Install development dependencies
    pip install -e ".[dev]"

Key Dependencies
--------------

The package relies on the following primary libraries:

* ``numpy`` and ``pandas`` for data manipulation
* ``backtesting`` for strategy backtesting
* ``matplotlib``, ``plotly``, and ``seaborn`` for visualization
* ``scikit-learn`` for machine learning components
* ``pyyaml`` for configuration management

Troubleshooting
--------------

Common installation issues:

1. **Python Version Error**: Make sure you have Python 3.11+ installed.

   .. code-block:: bash
   
       python --version

2. **Dependency Conflicts**: If you encounter package conflicts, try installing in a fresh virtual environment.

3. **Platform-Specific Issues**: Some packages may require additional system libraries depending on your OS. 