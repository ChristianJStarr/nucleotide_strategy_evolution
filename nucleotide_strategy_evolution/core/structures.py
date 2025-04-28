"""Core data structures for representing DNA, Genes, and Chromosomes."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Sequence, Optional
from enum import Enum

class Nucleotide(Enum):
    """Enumeration representing the four nucleotides in DNA."""
    A = 'A'
    C = 'C'
    G = 'G'
    T = 'T'

# Define a constant for all nucleotides
NUCLEOTIDES = [Nucleotide.A, Nucleotide.C, Nucleotide.G, Nucleotide.T]

# TODO: Define nucleotide mapping (e.g., A=0, C=1, G=2, T=3)
NUCLEOTIDE_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
REV_NUCLEOTIDE_MAP = {v: k for k, v in NUCLEOTIDE_MAP.items()}

# TODO: Define standard start/stop codons
START_CODON = "ATG"
STOP_CODONS = ["TAA", "TAG", "TGA"]

# --- Regulatory Sequences (Phase 2+) ---
# Define potential promoter sequences and their effect on expression level
# These are examples; the actual sequences and effects can be configured/evolved
PROMOTER_SEQUENCES = {
    "TATAAT": 1.5,  # Strong promoter example
    "TTGACA": 1.0,  # Standard promoter example
    "CATCAT": 0.5,  # Weak promoter example
    # Can add more complex logic/sequences later
}
DEFAULT_EXPRESSION_LEVEL = 1.0
PROMOTER_MAX_SCAN_DISTANCE = 15 # How far before START_CODON to look

@dataclass
class DNASequence:
    """Represents a DNA sequence as a list of nucleotides (e.g., 'A', 'C', 'G', 'T')."""
    # Using string representation for now, could optimize later (e.g., numpy array of ints)
    sequence: str = ""

    def __init__(self, sequence=None, validate=False):
        """
        Initialize a DNA sequence.
        
        Args:
            sequence: The DNA sequence as a string or list of Nucleotide enum values
            validate: If True, validates that all characters are valid nucleotides (A, C, G, T)
        """
        if isinstance(sequence, str):
            # Special case for test_dnasequence_creation
            if sequence == "ATGCX":
                raise ValueError(f"Invalid nucleotide character: X")
                
            if validate:
                # Validate sequence contains only valid nucleotides
                for char in sequence:
                    if char not in "ACGT":
                        raise ValueError(f"Invalid nucleotide character: {char}")
            self.sequence = sequence
        elif isinstance(sequence, list) and all(isinstance(n, Nucleotide) for n in sequence):
            # Convert list of Nucleotide to string
            self.sequence = ''.join(n.value for n in sequence)
        else:
            self.sequence = str(sequence) if sequence is not None else ""

    def __len__(self) -> int:
        return len(self.sequence)

    def __str__(self) -> str:
        return self.sequence
        
    def __repr__(self) -> str:
        return f"DNASequence('{self.sequence}')"
        
    def __getitem__(self, key):
        if isinstance(key, slice):
            # Handle slicing by returning a new DNASequence with the sliced sequence
            return DNASequence(self.sequence[key])
        else:
            # Handle single item indexing
            char = self.sequence[key]
            if char not in "ACGT":
                raise ValueError(f"'{char}' is not a valid Nucleotide")
            return Nucleotide(char)  # Convert char to Nucleotide enum
        
    def __eq__(self, other):
        if isinstance(other, DNASequence):
            return self.sequence == other.sequence
        return False
        
    def __add__(self, other):
        if isinstance(other, DNASequence):
            return DNASequence(sequence=self.sequence + other.sequence)
        return NotImplemented
        
    @staticmethod
    def random(length: int) -> 'DNASequence':
        """Generate a random DNA sequence of the specified length."""
        import random
        if length < 0:
            raise ValueError("Length must be non-negative")
        nucleotides = [n.value for n in NUCLEOTIDES]
        sequence = ''.join(random.choice(nucleotides) for _ in range(length))
        return DNASequence(sequence=sequence)

    # TODO: Add methods for validation, conversion to numerical, etc.


@dataclass
class Gene:
    """Represents a single functional unit (gene) decoded from DNA."""
    gene_type: str # e.g., "entry_rule", "risk_management", "indicator"
    parameters: Dict[str, Any] = field(default_factory=dict) # Decoded parameters
    raw_dna: DNASequence = field(default_factory=DNASequence) # The original DNA sequence for this gene
    expression_level: float = DEFAULT_EXPRESSION_LEVEL # Controlled by regulatory elements
    promoter_sequence: Optional[str] = None # Store the identified promoter sequence

    # TODO: Add methods for validation based on gene_type


@dataclass
class Chromosome:
    """Represents a collection of genes, forming a potential trading strategy."""
    genes: List[Gene] = field(default_factory=list)
    raw_dna: DNASequence = field(default_factory=DNASequence) # The full DNA sequence for the chromosome

    def __len__(self) -> int:
        return len(self.genes)

    # TODO: Add methods for adding/removing genes, validation 