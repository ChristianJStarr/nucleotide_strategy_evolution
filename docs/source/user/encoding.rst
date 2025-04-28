=================
DNA Encoding System
=================

This guide explains how trading strategies are encoded as DNA in the framework.

Basic Principles
--------------

In the Nucleotide Strategy Evolution framework, trading strategies are represented as DNA sequences consisting of the four nucleotides: **A**, **C**, **G**, and **T**. This genetic encoding allows strategies to:

1. **Evolve naturally** through mutation and crossover
2. **Express complex behaviors** through combinations of genes
3. **Support structural variation** including gene duplication, deletion, and rearrangement

DNA Structure Hierarchy
---------------------

.. code-block:: text

    DNA Sequence → Codons → Genes → Chromosome → Strategy

1. **Nucleotides** (A, C, G, T): The basic building blocks
2. **Codons**: Groups of 3 nucleotides (e.g., ATG, GCT) that encode specific values
3. **Genes**: Functional units starting with a start codon (ATG) and ending with a stop codon (TAA, TAG, TGA)
4. **Chromosome**: A complete DNA sequence containing multiple genes that together form a strategy

Codon to Value Mapping
--------------------

Each codon maps to a value in the range 0-63:

.. code-block:: python

    # Example: Convert codon to value
    def codon_to_value(codon):
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        return mapping[codon[0]] * 16 + mapping[codon[1]] * 4 + mapping[codon[2]]
    
    # ATG = 0*16 + 3*4 + 2 = 14

These values can then be scaled to the appropriate range for each parameter type.

Gene Structure
------------

Each gene follows this structure:

.. code-block:: text

    [Promoter Region] START_CODON [Type Codon] [Parameter Codons...] STOP_CODON

- **Promoter Region**: Optional regulatory sequence that affects gene expression
- **START_CODON**: Always `ATG`
- **Type Codon**: Determines the gene type (indicator, rule, risk management, etc.)
- **Parameter Codons**: Specific parameters for that gene type
- **STOP_CODON**: One of `TAA`, `TAG`, or `TGA`

Gene Types
---------

The framework supports various gene types:

1. **Indicator Genes**: Define technical indicators
   
   .. code-block:: text
   
       ATG [type=indicator] [indicator_name] [period] [source] ... TGA
   
2. **Rule Genes**: Define entry/exit conditions
   
   .. code-block:: text
   
       ATG [type=rule] [condition_type] [operator] [threshold] ... TAA
   
3. **Risk Management Genes**: Define risk parameters
   
   .. code-block:: text
   
       ATG [type=risk] [stop_loss_type] [stop_loss_value] [take_profit_type] ... TAG

4. **Filter Genes**: Define market conditions for strategy activation
   
   .. code-block:: text
   
       ATG [type=filter] [filter_type] [parameter1] [parameter2] ... TAA

Example DNA Sequence
------------------

.. code-block:: text

    ATGCGTTCTGATACGCTAGGCATTTAAAATGAGTTCGTGATCAATTGACGTAA...
    ^^^     ^^^^^^^^^^^^^^^^^^^^ ^^^     ^^^^^^^^^^^^^^^^^
    START   Parameters for RSI    STOP    Parameters for MA crossover
    
Working with DNA
--------------

The framework provides utilities to:

1. **Generate random DNA** for initialization:
   
   .. code-block:: python
   
       from nucleotide_strategy_evolution.core.structures import DNASequence
       
       # Generate random DNA of length 500
       dna = DNASequence.random(500)
   
2. **Decode DNA** into strategy components:
   
   .. code-block:: python
   
       from nucleotide_strategy_evolution.encoding import decode_chromosome
       
       # Decode DNA into chromosome
       chromosome = decode_chromosome(dna)
       
       # Access genes
       for gene in chromosome.genes:
           print(f"Gene type: {gene.type}, Parameters: {gene.parameters}")
   
3. **Encode strategy components** back to DNA:
   
   .. code-block:: python
   
       from nucleotide_strategy_evolution.encoding import encode_chromosome
       
       # Encode chromosome back to DNA
       modified_dna = encode_chromosome(chromosome)

See the :doc:`API reference <../api/encoding>` for more details on encoding functions. 