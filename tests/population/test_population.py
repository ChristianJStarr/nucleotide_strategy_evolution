"""Tests for the Population class."""

import pytest
from unittest.mock import MagicMock, Mock, patch

from nucleotide_strategy_evolution.population.population import Population, DEFAULT_DNA_LENGTH
from nucleotide_strategy_evolution.core.structures import Chromosome, DNASequence
from nucleotide_strategy_evolution.fitness.evaluation import BasicFitnessEvaluator, MultiObjectiveEvaluator, FitnessType

# --- Test Initialization ---

def test_population_init_valid():
    pop = Population(size=10, dna_length=50)
    assert pop.target_size == 10
    assert pop.dna_length == 50
    assert len(pop.individuals) == 0
    assert len(pop.fitness_scores) == 0

def test_population_init_default_dna_length():
    pop = Population(size=20)
    assert pop.target_size == 20
    assert pop.dna_length == DEFAULT_DNA_LENGTH

@pytest.mark.parametrize("size", [0, -1, -10])
def test_population_init_invalid_size(size: int):
    with pytest.raises(ValueError, match="Population size must be positive"):
        Population(size=size)

@pytest.mark.parametrize("dna_length", [0, -1, -50])
def test_population_init_invalid_dna_length(dna_length: int):
    with pytest.raises(ValueError, match="DNA length must be positive"):
        Population(size=10, dna_length=dna_length)

# --- Test Population Initialization (Populating) ---

@patch('nucleotide_strategy_evolution.population.population.decode_chromosome')
@patch('nucleotide_strategy_evolution.population.population.random_dna_sequence')
def test_population_initialize(mock_random_dna, mock_decode_chromo):
    pop_size = 15
    dna_len = 60
    # Configure mocks
    mock_dna = DNASequence("A" * dna_len)
    mock_chromo = Chromosome(raw_dna=mock_dna)
    mock_random_dna.return_value = mock_dna
    mock_decode_chromo.return_value = mock_chromo
    
    pop = Population(size=pop_size, dna_length=dna_len)
    pop.initialize()
    
    assert len(pop) == pop_size
    assert len(pop.individuals) == pop_size
    assert len(pop.fitness_scores) == pop_size
    assert mock_random_dna.call_count == pop_size
    assert mock_decode_chromo.call_count == pop_size
    # Check if all individuals are the mocked chromosome
    assert all(ind is mock_chromo for ind in pop.individuals)
    # Check if fitness scores are initialized (to None in current implementation)
    assert all(pop.fitness_scores[i] is None for i in range(pop_size))

# --- Test Accessors and Length ---

def test_population_len():
    pop = Population(size=5)
    assert len(pop) == 0 # Before initialization
    pop.individuals = [MagicMock(Chromosome)] * 3 # Manually add individuals
    assert len(pop) == 3
    pop.initialize()
    assert len(pop) == 5
    
def test_population_getitem():
    pop = Population(size=3)
    chromo1 = Chromosome(raw_dna=DNASequence("AAA"))
    chromo2 = Chromosome(raw_dna=DNASequence("CCC"))
    pop.individuals = [chromo1, chromo2]
    assert pop[0] is chromo1
    assert pop[1] is chromo2
    with pytest.raises(IndexError):
        _ = pop[2]

def test_population_get_set_fitness():
    pop = Population(size=3)
    pop.initialize() # Need individuals to set fitness for
    assert pop.get_fitness(0) is None
    assert pop.get_fitness(1) is None
    
    fitness1: FitnessType = (10.0, -0.1)
    pop.set_fitness(0, fitness1)
    assert pop.get_fitness(0) == fitness1
    assert pop.fitness_scores[0] == fitness1
    
    fitness2 = (5.0, -0.5)
    pop.set_fitness(1, fitness2)
    assert pop.get_fitness(1) == fitness2
    assert pop.fitness_scores[1] == fitness2
    
    # Test getting unset fitness
    assert pop.get_fitness(2) is None
    # Test setting out of bounds
    with pytest.raises(IndexError):
        pop.set_fitness(3, (1.0, 1.0))
    # Test getting invalid index
    assert pop.get_fitness(10) is None

# --- Test Fitness Evaluation (Sequential in Population class) ---

def test_evaluate_fitnesses_basic():
    pop = Population(size=3)
    pop.initialize()
    
    # Mock the evaluator and backtester
    mock_evaluator = MagicMock(spec=BasicFitnessEvaluator)
    mock_evaluator.evaluate.side_effect = [10.5, -2.3, 100.0] # Return different values
    
    mock_rules = {"rule1": True}
    pop.evaluate_fitnesses(mock_evaluator, mock_rules)
    
    assert mock_evaluator.evaluate.call_count == 3
    # Check if evaluate was called with each chromosome and the rules
    mock_evaluator.evaluate.assert_any_call(pop[0], mock_rules)
    mock_evaluator.evaluate.assert_any_call(pop[1], mock_rules)
    mock_evaluator.evaluate.assert_any_call(pop[2], mock_rules)
    
    # Check if fitness scores were stored
    assert pop.get_fitness(0) == 10.5
    assert pop.get_fitness(1) == -2.3
    assert pop.get_fitness(2) == 100.0

def test_evaluate_fitnesses_moo():
    pop = Population(size=2)
    pop.initialize()
    
    # Mock the MOO evaluator
    mock_evaluator = MagicMock(spec=MultiObjectiveEvaluator)
    fit1: FitnessType = (10.0, -0.1)
    fit2: FitnessType = (20.0, -0.05)
    mock_evaluator.evaluate.side_effect = [fit1, fit2]
    # Mock attributes needed for error handling
    mock_evaluator.objectives = ['profit', '-drawdown']
    mock_evaluator.parsed_objectives = [('profit', False), ('drawdown', True)]
    
    mock_rules = {}
    pop.evaluate_fitnesses(mock_evaluator, mock_rules)
    
    assert mock_evaluator.evaluate.call_count == 2
    assert pop.get_fitness(0) == fit1
    assert pop.get_fitness(1) == fit2
    
def test_evaluate_fitnesses_eval_error():
    pop = Population(size=2)
    pop.initialize()
    
    mock_evaluator = MagicMock(spec=MultiObjectiveEvaluator)
    fit1 = (10.0, -0.1)
    # Simulate an error during evaluation of the second individual
    mock_evaluator.evaluate.side_effect = [fit1, Exception("Backtest failed")]
    # Mock attributes needed for error handling
    mock_evaluator.objectives = ['profit', '-drawdown']
    mock_evaluator.parsed_objectives = [('profit', False), ('drawdown', True)]
    
    mock_rules = {}
    pop.evaluate_fitnesses(mock_evaluator, mock_rules)
    
    assert mock_evaluator.evaluate.call_count == 2
    assert pop.get_fitness(0) == fit1
    # Check if worst fitness was assigned on error
    worst_fitness = (-float('inf'), float('inf'))
    assert pop.get_fitness(1) == worst_fitness

def test_evaluate_fitnesses_unsupported_evaluator():
    pop = Population(size=2)
    pop.initialize()
    mock_evaluator = Mock() # Not an instance of Basic or MultiObjective
    with pytest.raises(TypeError, match="Unsupported fitness evaluator type"):
        pop.evaluate_fitnesses(mock_evaluator, {})

# TODO: Add tests for initialization with template
# TODO: Add tests for evaluation error handling (e.g., evaluator fails for one individual) 