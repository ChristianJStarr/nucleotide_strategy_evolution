==========
Quickstart
==========

This guide will help you quickly get started with the Nucleotide Strategy Evolution framework.

Basic Example
------------

Here's a minimal example showing how to use the framework:

.. code-block:: python

    from nucleotide_strategy_evolution.population import Population
    from nucleotide_strategy_evolution.fitness import MultiObjectiveEvaluator
    from nucleotide_strategy_evolution.operators import get_crossover_operator, get_mutation_operator
    from nucleotide_strategy_evolution.population import get_selection_operator
    from nucleotide_strategy_evolution.backtesting import setup_backtester

    # 1. Configure evolution parameters
    evo_params = {
        'population': {'size': 50, 'dna_length': 500},
        'evolution': {'generations': 20},
        'fitness': {'objectives': ['net_profit', '-max_drawdown']},
        'operators': {
            'crossover': {'type': 'single_point', 'rate': 0.7},
            'mutations': [
                {'type': 'point_mutation', 'rate': 0.05},
                {'type': 'insertion', 'rate': 0.01},
                {'type': 'deletion', 'rate': 0.01}
            ]
        },
        'selection': {'method': 'nsga2'}
    }
    
    # 2. Setup backtester with market data
    backtester = setup_backtester("your_market_data.csv", initial_cash=100000)
    
    # 3. Initialize population
    population = Population(
        size=evo_params['population']['size'], 
        dna_length=evo_params['population']['dna_length']
    )
    population.initialize()
    
    # 4. Setup evaluator and genetic operators
    fitness_evaluator = MultiObjectiveEvaluator(
        backtester, 
        objectives=evo_params['fitness']['objectives']
    )
    crossover_op = get_crossover_operator(evo_params['operators']['crossover'])
    mutation_ops = [
        (get_mutation_operator(m_conf), m_conf['rate']) 
        for m_conf in evo_params['operators']['mutations']
    ]
    selection_method = get_selection_operator(evo_params['selection'])
    
    # 5. Run evolution for specified generations
    for gen in range(evo_params['evolution']['generations']):
        print(f"Generation {gen+1}/{evo_params['evolution']['generations']}")
        
        # Evaluate all strategies in the population
        fitness_evaluator.evaluate_population(population)
        
        # Create next generation through selection, crossover, and mutation
        population.evolve(selection_method, crossover_op, mutation_ops)
    
    # 6. Get the best strategies
    best_strategies = population.get_elite(n=5)
    
    # 7. Analyze the results
    for i, strategy in enumerate(best_strategies):
        print(f"Strategy {i+1}: {strategy.fitness}")

Configuration with YAML
----------------------

For more complex setups, you can use YAML configuration files:

.. code-block:: yaml

    # config/simple_evolution.yaml
    population:
      size: 100
      dna_length: 1000
    
    evolution:
      generations: 50
      
    fitness:
      objectives: 
        - net_profit
        - "-max_drawdown"
        - sharpe_ratio
      
    operators:
      crossover:
        type: single_point
        rate: 0.7
      mutations:
        - type: point_mutation
          rate: 0.05
        - type: insertion
          rate: 0.01
        - type: gene_duplication
          rate: 0.01
          
    selection:
      method: nsga2

Then load the configuration in your script:

.. code-block:: python

    from nucleotide_strategy_evolution.utils import config_loader
    
    config = config_loader.load_config("config/simple_evolution.yaml")
    
    # Use the configuration
    population = Population(
        size=config['population']['size'],
        dna_length=config['population']['dna_length']
    )

Next Steps
---------

* Check out the full :doc:`examples <examples/basic>` for more usage patterns
* Learn about the :doc:`core concepts <concepts>` of genetic algorithms and trading strategy evolution
* Explore the :doc:`API reference <api/core>` for detailed documentation of all classes and functions 