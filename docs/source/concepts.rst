=======
Concepts
=======

This page explains the key concepts and design principles behind the Nucleotide Strategy Evolution framework.

Genetic Algorithms Primer
------------------------

Genetic algorithms (GAs) are search and optimization methods inspired by natural evolution. The key components are:

1. **Population**: A set of candidate solutions (individuals)
2. **Selection**: Process of choosing individuals for reproduction based on fitness
3. **Crossover (Recombination)**: Exchange of genetic material between parents to create offspring
4. **Mutation**: Small random changes in offspring to maintain genetic diversity
5. **Fitness Function**: Evaluates how good a solution is
6. **Generations**: The iterative process of evolving from one population to the next

DNA Representation
----------------

In this framework, trading strategies are encoded as DNA sequences:

.. code-block:: text

    ATGCGTACGTCGATCGATCGTACGTAGCATTCGACTGATGCTA...

DNA Sequence Structure
~~~~~~~~~~~~~~~~~~~~~

- **Nucleotides**: The four basic units (A, C, G, T)
- **Codons**: Triplets of nucleotides (e.g., ATG, GCT) that map to values
- **Genes**: Functional blocks starting with a start codon (ATG) and ending with stop codons (TAA, TAG, TGA)
- **Chromosomes**: Collections of genes that together form a complete strategy

Gene Types
~~~~~~~~~

Different genes control different aspects of trading strategies:

- **Indicator Genes**: Technical indicators (e.g., Moving Averages, RSI)
- **Rule Genes**: Entry and exit conditions
- **Risk Management Genes**: Stop-loss, take-profit, position sizing
- **Filter Genes**: Time-based or market regime filters
- **Order Type Genes**: Market, limit, or other order types

Genetic Operators
---------------

The framework uses various genetic operators:

Mutation Types
~~~~~~~~~~~~

- **Point Mutation**: Changes a single nucleotide (A→C, G→T, etc.)
- **Insertion Mutation**: Adds one or more nucleotides
- **Deletion Mutation**: Removes one or more nucleotides
- **Gene Duplication**: Copies a gene and inserts it elsewhere
- **Codon Inversion**: Reverses the order of nucleotides in a codon
- **Translocation**: Moves a section of DNA to another location

Crossover Types
~~~~~~~~~~~~~

- **Single-Point Crossover**: Exchanges DNA at a single point
- **Uniform Crossover**: Exchanges individual nucleotides with some probability
- **Multi-Point Crossover**: Exchanges DNA at multiple points

Fitness Evaluation
----------------

Strategies are evaluated on multiple objectives:

- **Profitability**: Net profit, return on investment
- **Risk Management**: Maximum drawdown, Sortino ratio, Sharpe ratio
- **Trade Metrics**: Win rate, profit factor, average trade
- **Robustness**: Performance consistency across market regimes

Selection Methods
--------------

The framework supports various selection strategies:

- **Tournament Selection**: Selects the best from random subgroups
- **NSGA-II Selection**: Non-dominated sorting for multi-objective optimization
- **Lexicase Selection**: Evaluates candidates on objectives in random order

Diversity Preservation
--------------------

Techniques to maintain strategy diversity:

- **Fitness Sharing**: Penalizes similar individuals to maintain genetic diversity
- **Novelty Search**: Rewards unique behavioral characteristics
- **MAP-Elites**: Maintains an archive of strategies across different behavioral dimensions

Validation Methods
---------------

Methods to combat overfitting:

- **Walk-Forward Optimization (WFO)**: Tests strategies on consecutive, non-overlapping periods
- **Purged K-Fold Cross-Validation**: Modified CV that respects time series data

Framework Architecture
-------------------

The framework is organized into modules:

- **core**: Basic data structures
- **encoding**: DNA-to-strategy mapping
- **operators**: Genetic operators
- **population**: Population management
- **fitness**: Evaluation methods
- **backtesting**: Strategy testing
- **analysis**: Results analysis
- **visualization**: Data visualization 