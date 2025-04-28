"""Tests for core data structures."""

import pytest
from typing import List, Any

from nucleotide_strategy_evolution.core.structures import (
    DNASequence, Nucleotide, Gene, Chromosome,
    START_CODON, STOP_CODONS
)

def test_dnasequence_creation():
    """Test creating DNASequence objects."""
    seq_str = "ATGCGT"
    seq = DNASequence(seq_str)
    assert seq.sequence == seq_str
    assert len(seq) == len(seq_str)
    assert str(seq) == seq_str

    # Test creation with list of Nucleotides
    nucleotides = [Nucleotide.A, Nucleotide.T, Nucleotide.G, Nucleotide.C]
    seq_from_list = DNASequence(nucleotides)
    assert seq_from_list.sequence == "ATGC"
    assert len(seq_from_list) == 4

    # Test creation with invalid characters
    with pytest.raises(ValueError):
        DNASequence("ATGCX")

    # Test empty sequence
    empty_seq = DNASequence("")
    assert len(empty_seq) == 0
    assert empty_seq.sequence == ""

def test_dnasequence_equality():
    """Test equality comparisons for DNASequence."""
    seq1 = DNASequence("ATGC")
    seq2 = DNASequence("ATGC")
    seq3 = DNASequence("ATGG")
    seq4 = DNASequence("TAGC")
    assert seq1 == seq2
    assert seq1 != seq3
    assert seq1 != seq4
    assert seq1 != "ATGC" # Should not be equal to string

def test_dnasequence_getitem():
    """Test accessing elements by index."""
    seq = DNASequence("GATTACA")
    assert seq[0] == Nucleotide.G
    assert seq[1] == Nucleotide.A
    assert seq[-1] == Nucleotide.A
    assert seq[-2] == Nucleotide.C

    # Test slicing
    slice1 = seq[1:4] # ATT
    assert isinstance(slice1, DNASequence)
    assert slice1.sequence == "ATT"

    slice2 = seq[:3] # GAT
    assert slice2.sequence == "GAT"

    slice3 = seq[4:] # ACA
    assert slice3.sequence == "ACA"
    
    slice4 = seq[:] # Full copy
    assert slice4.sequence == "GATTACA"
    assert slice4 is not seq # Should be a new object

    # Test index out of bounds
    with pytest.raises(IndexError):
        _ = seq[10]
        
def test_dnasequence_concatenation():
    """Test concatenating DNASequence objects."""
    seq1 = DNASequence("ATGC")
    seq2 = DNASequence("CGTA")
    seq3 = DNASequence("AATT")
    
    combined12 = seq1 + seq2
    assert isinstance(combined12, DNASequence)
    assert combined12.sequence == "ATGCCGTA"
    assert len(combined12) == 8
    
    combined123 = seq1 + seq2 + seq3
    assert combined123.sequence == "ATGCCGTAAATT"
    assert len(combined123) == 12
    
    # Test adding empty sequence
    combined1_empty = seq1 + DNASequence("")
    assert combined1_empty.sequence == seq1.sequence
    assert combined1_empty is not seq1
    
    empty_combined1 = DNASequence("") + seq1
    assert empty_combined1.sequence == seq1.sequence
    assert empty_combined1 is not seq1

def test_dnasequence_repr():
    """Test the repr output."""
    seq = DNASequence("ACGT")
    assert repr(seq) == "DNASequence('ACGT')"
    
    empty_seq = DNASequence("")
    assert repr(empty_seq) == "DNASequence('')"

def test_dnasequence_random():
     """Test the random sequence generation class method."""
     length = 50
     rand_seq = DNASequence.random(length)
     assert isinstance(rand_seq, DNASequence)
     assert len(rand_seq) == length
     # Check if all nucleotides are valid
     valid_nucleotides = {n.value for n in Nucleotide}
     for char in rand_seq.sequence:
         assert char in valid_nucleotides
         
     rand_seq_zero = DNASequence.random(0)
     assert len(rand_seq_zero) == 0
     assert rand_seq_zero.sequence == ""
     
     with pytest.raises(ValueError):
         DNASequence.random(-1)

# --- Tests for Chromosome ---

def test_chromosome_creation():
    """Test creating Chromosome objects."""
    dna = DNASequence("ATGCGT")
    gene1 = Gene(gene_type="entry", parameters={"p1": 10}, raw_dna=DNASequence("ATG"))
    gene2 = Gene(gene_type="exit", parameters={"p2": 20}, raw_dna=DNASequence("CGT"))
    
    # Create with list of genes and raw DNA
    chromo = Chromosome(genes=[gene1, gene2], raw_dna=dna)
    assert len(chromo) == 2
    assert chromo.genes[0] == gene1
    assert chromo.genes[1] == gene2
    assert chromo.raw_dna == dna

    # Create empty chromosome
    empty_chromo = Chromosome()
    assert len(empty_chromo) == 0
    assert len(empty_chromo.genes) == 0
    assert len(empty_chromo.raw_dna) == 0

    # Create with only raw DNA
    dna_only_chromo = Chromosome(raw_dna=dna)
    assert len(dna_only_chromo) == 0
    assert len(dna_only_chromo.genes) == 0
    assert dna_only_chromo.raw_dna == dna

def test_chromosome_length():
    """Test the __len__ method of Chromosome."""
    gene1 = Gene(gene_type="A")
    gene2 = Gene(gene_type="B")
    chromo1 = Chromosome(genes=[gene1, gene2])
    assert len(chromo1) == 2
    
    chromo2 = Chromosome(genes=[gene1])
    assert len(chromo2) == 1
    
    empty_chromo = Chromosome()
    assert len(empty_chromo) == 0

# Add more tests here as Chromosome class gets more methods (e.g., add_gene, get_gene_by_type) 


# --- Tests for Encoding/Decoding Functions (Moved from tests/encoding) ---

from nucleotide_strategy_evolution.encoding import (
    dna_to_int,
    int_to_dna,
    scale_value,
    scale_value_int,
    map_value_to_choices,
    find_genes_in_dna,
    decode_gene,
    decode_chromosome,
    encode_gene,
    encode_chromosome,
    random_dna_sequence
)


# --- Test Codon/Int Conversions ---

@pytest.mark.parametrize("dna, expected_int", [
    ("AAA", 0),
    ("AAC", 1),
    ("AAG", 2),
    ("AAT", 3),
    ("ACA", 4),
    ("AGA", 8),
    ("ATA", 12),
    ("CAA", 16),
    ("GAA", 32),
    ("TAA", 48),
    ("CCC", 21),
    ("GGG", 42),
    ("TTT", 63),
    ("GCT", 39),
])
def test_dna_to_int(dna: str, expected_int: int):
    assert dna_to_int(dna) == expected_int

@pytest.mark.parametrize("value", ["AA", "AATT", "ABC"])
def test_dna_to_int_invalid_length(value: str):
    with pytest.raises(ValueError):
        dna_to_int(value)

def test_dna_to_int_invalid_nucleotide():
    with pytest.raises(ValueError):
        dna_to_int("AXT")

@pytest.mark.parametrize("value, expected_dna", [
    (0, "AAA"),
    (1, "AAC"),
    (2, "AAG"),
    (3, "AAT"),
    (4, "ACA"),
    (8, "AGA"),
    (12, "ATA"),
    (16, "CAA"),
    (32, "GAA"),
    (48, "TAA"),
    (21, "CCC"),
    (42, "GGG"),
    (63, "TTT"),
    (39, "GCT"),
])
def test_int_to_dna(value: int, expected_dna: str):
    assert int_to_dna(value) == expected_dna

@pytest.mark.parametrize("value", [-1, 64, 100])
def test_int_to_dna_invalid_value(value: int):
    with pytest.raises(ValueError):
        int_to_dna(value)

# --- Test Random DNA Sequence ---
def test_random_dna_sequence_encoding(): # Renamed to avoid conflict
     """Test the random sequence generation (from encoding module)."""
     length = 50
     rand_seq = random_dna_sequence(length)
     assert isinstance(rand_seq, DNASequence)
     assert len(rand_seq) == length
     # Check if all nucleotides are valid
     valid_nucleotides_chars = {'A', 'C', 'G', 'T'}
     for char in rand_seq.sequence:
         assert char in valid_nucleotides_chars

     rand_seq_zero = random_dna_sequence(0)
     assert len(rand_seq_zero) == 0
     assert rand_seq_zero.sequence == ""

     with pytest.raises(ValueError):
         random_dna_sequence(-1)

# --- Test Scaling/Mapping Helpers ---

@pytest.mark.parametrize("codon_val, min_val, max_val, expected", [
    (0, 0.0, 100.0, 0.0),
    (63, 0.0, 100.0, 100.0),
    (31, 0.0, 100.0, pytest.approx(49.2063)),
    (0, -10.0, 10.0, -10.0),
    (63, -10.0, 10.0, 10.0),
    (15, 1.0, 5.0, pytest.approx(1.9523)),
])
def test_scale_value(codon_val: int, min_val: float, max_val: float, expected: float):
    assert scale_value(codon_val, min_val, max_val) == expected

@pytest.mark.parametrize("codon_val, min_val, max_val, expected", [
    (0, 0, 100, 0),
    (63, 0, 100, 100),
    (31, 0, 100, 49),
    (0, -10, 10, -10),
    (63, -10, 10, 10),
    (15, 1, 5, 2),
    (45, 10, 20, 17),
])
def test_scale_value_int(codon_val: int, min_val: int, max_val: int, expected: int):
    assert scale_value_int(codon_val, min_val, max_val) == expected

@pytest.mark.parametrize("codon_val, choices, expected", [
    (0, ["A", "B", "C"], "A"),
    (20, ["A", "B", "C"], "A"),
    (21, ["A", "B", "C"], "B"),
    (42, ["A", "B", "C"], "B"),
    (43, ["A", "B", "C"], "C"),
    (63, ["A", "B", "C"], "C"),
    (0, [10, 20, 30, 40], 10),
    (15, [10, 20, 30, 40], 10),
    (16, [10, 20, 30, 40], 20),
    (31, [10, 20, 30, 40], 20),
    (32, [10, 20, 30, 40], 30),
    (63, [10, 20, 30, 40], 40),
    (30, [True, False], True),
    (32, [True, False], False),
])
def test_map_value_to_choices(codon_val: int, choices: List[Any], expected: Any):
    assert map_value_to_choices(codon_val, choices) == expected

def test_map_value_to_choices_empty():
    assert map_value_to_choices(30, []) is None

# --- Tests for Gene Finding ---

def test_find_genes_no_start():
    dna = DNASequence("AAACCCTTTGGGTAATAGTAG")
    assert find_genes_in_dna(dna) == []

def test_find_genes_start_no_stop():
    dna = DNASequence("AAACCCTTTATGCCCGGGTTT")
    assert find_genes_in_dna(dna) == []

def test_find_genes_simple():
    dna = DNASequence("XXATGAAACCCGGGXXXTAGYYY")
    expected_start = 2
    expected_end = 20
    expected_gene = "ATGAAACCCGGGXXXTAG"
    genes = find_genes_in_dna(dna)
    assert len(genes) == 1
    start, end, raw, promo = genes[0]
    assert start == expected_start
    assert end == expected_end
    assert raw == expected_gene
    assert promo is None

def test_find_genes_multiple():
    dna = DNASequence("XXATGAAACCCGGGTAGYYYYATGTTTAAATAAZZZ")
    genes = find_genes_in_dna(dna)
    assert len(genes) == 2
    assert genes[0] == (2, 17, "ATGAAACCCGGGTAG", None)
    assert genes[1] == (21, 33, "ATGTTTAAATAA", None)

def test_find_genes_overlapping_starts():
    dna = DNASequence("XXATGATGAAACCCTAGYYY")
    genes = find_genes_in_dna(dna)
    assert len(genes) == 1
    assert genes[0] == (2, 17, "ATGATGAAACCCTAG", None)

def test_find_genes_with_promoter():
    promoter_seq = "TATAAT"
    dna = DNASequence(f"XX{promoter_seq}YYYATGAAACCCGGGTAGZZZ")
    genes = find_genes_in_dna(dna)
    assert len(genes) == 1
    assert genes[0] == (11, 26, "ATGAAACCCGGGTAG", promoter_seq)

def test_find_genes_promoter_too_far():
    promoter_seq = "TATAAT"
    dna = DNASequence(f"XX{promoter_seq}YYYYYYYYYYYYYYYATGAAACCCGGGTAGZZZ")
    genes = find_genes_in_dna(dna)
    assert len(genes) == 1
    assert genes[0] == (26, 41, "ATGAAACCCGGGTAG", None)

# --- Tests for Gene Decoding ---

def test_decode_gene_basic_types():
    gene_entry = decode_gene(START_CODON + int_to_dna(5) + "AAAAAA" + STOP_CODONS[0])
    assert gene_entry.gene_type == "entry_rule"
    gene_exit = decode_gene(START_CODON + int_to_dna(20) + "CCCCCC" + STOP_CODONS[1], promoter="CATCAT")
    assert gene_exit.gene_type == "exit_rule"
    assert gene_exit.expression_level == 0.5
    gene_risk = decode_gene(START_CODON + int_to_dna(40) + "GGGGGG" + STOP_CODONS[2])
    assert gene_risk.gene_type == "risk_management"
    gene_indic = decode_gene(START_CODON + int_to_dna(55) + "TTTTTT" + STOP_CODONS[0])
    assert gene_indic.gene_type == "indicator"
    gene_time = decode_gene(START_CODON + int_to_dna(58) + "ACGTAC" + STOP_CODONS[1])
    assert gene_time.gene_type == "time_filter"
    gene_order = decode_gene(START_CODON + int_to_dna(61) + "TGCACT" + STOP_CODONS[2])
    assert gene_order.gene_type == "order_management"
    gene_regime = decode_gene(START_CODON + int_to_dna(62) + "GGTACC" + STOP_CODONS[0])
    assert gene_regime.gene_type == "regime_filter"
    gene_meta = decode_gene(START_CODON + int_to_dna(63) + "CCAAGG" + STOP_CODONS[1])
    assert gene_meta.gene_type == "meta"

def test_decode_gene_invalid():
    assert decode_gene(START_CODON + STOP_CODONS[0]) is None
    assert decode_gene("AAACCCGGGTTT") is None
    gene_invalid_type = decode_gene(START_CODON + "XYZ" + "AAACCC" + STOP_CODONS[0])
    assert gene_invalid_type.gene_type == "unknown"

def test_decode_gene_indicator_params():
    rsi_dna_str = START_CODON + int_to_dna(48) + int_to_dna(0) + int_to_dna(15) + int_to_dna(5) + STOP_CODONS[0]
    gene_rsi = decode_gene(rsi_dna_str)
    assert gene_rsi.parameters.get('indicator_name') == "SMA"
    assert gene_rsi.parameters.get('period') == 14
    assert gene_rsi.parameters.get('source') == "close"

    bb_dna_str = START_CODON + int_to_dna(50) + int_to_dna(50) + int_to_dna(21) + int_to_dna(31) + STOP_CODONS[1]
    gene_bb = decode_gene(bb_dna_str)
    assert gene_bb.parameters.get('indicator_name') == "BBANDS"
    assert gene_bb.parameters.get('period') == 21
    assert gene_bb.parameters.get('std_dev') == pytest.approx(2.0)

# --- Tests for Chromosome Decoding ---

def test_decode_chromosome():
    promoter_seq = "TATAAT"
    gene1_dna_str = START_CODON + int_to_dna(5) + "AAAAAA" + STOP_CODONS[0]
    gene2_dna_str = START_CODON + int_to_dna(40) + "GGGGGG" + STOP_CODONS[2]
    full_dna_str = f"XXX{promoter_seq}YYY{gene1_dna_str}ZZZ{gene2_dna_str}WWW"
    full_dna = DNASequence(full_dna_str)
    chromosome = decode_chromosome(full_dna)
    assert len(chromosome.genes) == 2
    assert chromosome.raw_dna == full_dna
    assert chromosome.genes[0].gene_type == "entry_rule"
    assert chromosome.genes[0].promoter_sequence == promoter_seq
    assert chromosome.genes[0].expression_level == 1.5
    assert chromosome.genes[1].gene_type == "risk_management"
    assert chromosome.genes[1].promoter_sequence is None
    assert chromosome.genes[1].expression_level == 1.0

def test_decode_chromosome_empty():
    chromosome = decode_chromosome(DNASequence("ACGTACGTACGT"))
    assert len(chromosome.genes) == 0
    assert chromosome.raw_dna.sequence == "ACGTACGTACGT"

# --- Tests for Encoding (Basic) ---

def test_encode_gene_basic():
    raw = DNASequence(START_CODON + "AAACCC" + STOP_CODONS[0])
    gene = Gene(gene_type="entry", raw_dna=raw)
    assert encode_gene(gene) == raw
    gene_no_raw = Gene(gene_type="exit")
    assert encode_gene(gene_no_raw).sequence == ""

def test_encode_chromosome_basic():
    raw1 = DNASequence(START_CODON + "AAA" + STOP_CODONS[0])
    raw2 = DNASequence(START_CODON + "CCC" + STOP_CODONS[1])
    gene1 = Gene(gene_type="entry", raw_dna=raw1)
    gene2 = Gene(gene_type="exit", raw_dna=raw2)
    chromosome = Chromosome(genes=[gene1, gene2], raw_dna=DNASequence("IGNORED_FOR_NOW"))
    expected_dna = DNASequence(raw1.sequence + raw2.sequence)
    assert encode_chromosome(chromosome) == expected_dna
    empty_chromosome = Chromosome()
    assert encode_chromosome(empty_chromosome).sequence == "" 